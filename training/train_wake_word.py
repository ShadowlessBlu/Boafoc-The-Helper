"""Train an openWakeWord-compatible model for a custom wake word, on the Pi (CPU only).

Why: openWakeWord ships only English phrases, and the quick "record it 5 times" method could not
tell the user's words from ordinary speech (2026-11-19). This follows openWakeWord's own recipe,
scaled down to run on the Pi in under an hour:

  positives   the word, said by many of the 904 speakers of Piper's en_US-libritts_r voice, with
              phonetic spellings so it sounds right ("Bwah-foh" for Boafo), at varied speed and
              pitch, placed so it ends near the end of a 2 s window (that's when the detector
              should fire), mixed with real fan/room noise recorded from the assistant's mic
  negatives   ordinary sentences (many speakers), similar-sounding words, the other wake words,
              and room noise alone
  features    openWakeWord's own feature model (16 x 96 per 2 s), so the result runs in the
              existing detector unchanged
  model       a small neural network (scikit-learn), saved as ONNX with input (1, 16, 96) and one
              output score, the same shape as openWakeWord's "hey_jarvis"

  .venv/bin/python -u training/train_wake_word.py boafo [--positives 2000] [--work ~/wake-train]

Writes models/wake/<word>.onnx and <work>/<word>_report.json. Common negatives are cached in
<work>/cache, so the second and third word are faster.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
WIN = 2 * SR

SPELLINGS = {
    "uzi": ["Uzi", "Oozy", "Oo-zee", "Uzi!", "Oozie"],
    "aiya": ["Aiya", "Eye-ya", "Ayya", "Ai-yah", "Aiyah"],
    "boafo": ["Boafo", "Bwah-foh", "Boa-foh", "Bo-ah-fo", "Bwafo"],
}
CONFUSABLE = {
    "uzi": ["fuzzy", "busy", "Lucy", "juicy", "music", "you see", "use it", "Susie", "cozy", "who is he"],
    "aiya": ["fire", "higher", "liar", "Maya", "I am", "papaya", "hire ya", "I see", "eye", "buyer", "Iowa"],
    "boafo": ["buffalo", "both of", "go for", "sofa", "before", "boa", "bonfire", "photo", "Bova", "go far", "low fat"],
}
SENTENCES = """What time is it?
What's the weather like today?
Turn on the kitchen light.
Can you remind me to call mum on Sunday?
I'm going to the market later.
Please play some music.
How far is Kumasi from Accra?
The meeting starts at ten o'clock.
Did you see the football match last night?
I need to buy bread and eggs.
Set a timer for five minutes.
What is the capital of Ghana?
Tell me a short joke.
My sister's birthday is in March.
Switch off the bedroom light please.
How are you doing this morning?
Let's go outside for a walk.
The power went off again.
Can you hear me clearly?
I will be back in an hour.
Read me the news headlines.
What should I cook for dinner?
The car needs fuel before we leave.
Open the window, it's hot in here.
Remember that my password manager is on the laptop.
Who won the election?
Give me one tip for staying focused.
How many days until Christmas?
I'm tired, I think I'll sleep early.
Call me when you get home.
Where did I put my keys?
This rice is really good.
We should fix the fan this weekend.
Is it going to rain tomorrow?
Translate hello into French.
Add milk to the shopping list.
How do I reset the router?
The children are playing outside.
Thank you very much.
Good night, see you tomorrow.""".splitlines()
# Never used for training: the false-trigger stream test speaks these (v2 tested on training
# sentences, which flattered the result).
TEST_SENTENCES = """Could you check if the gate is locked?
I bought a new phone yesterday afternoon.
The bus to the city leaves every half hour.
How much does a kilo of tomatoes cost now?
She said the exam results come out next week.
We watched a film about the ocean last night.
Please send the report before Friday morning.
My brother is learning to play the guitar.
The library closes early on public holidays.
Is there any water left in the fridge?
Turn the volume down a little, please.
I think the battery is almost empty.
They are building a new road near the school.
What did the doctor say about your back?
Let me know when the food is ready.
Grandma is visiting us for the weekend.
The printer is out of paper again.
Which team plays first in the tournament?
I left my umbrella at the office.
Can we move the meeting to three o'clock?""".splitlines()


def resample(x: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import resample_poly
    if sr == SR:
        return x
    g = np.gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g)


def trim(x: np.ndarray) -> np.ndarray:
    """Cut leading/trailing near-silence from synthetic speech."""
    loud = np.flatnonzero(np.abs(x) > 0.02 * (np.abs(x).max() + 1e-9))
    return x[loud[0]:loud[-1] + 1] if len(loud) else x


class Synth:
    def __init__(self, voice_path: Path):
        from piper import PiperVoice
        self.voice = PiperVoice.load(str(voice_path))
        self.sr = self.voice.config.sample_rate
        self.speakers = self.voice.config.num_speakers

    def say(self, text: str, rng: np.random.Generator) -> np.ndarray:
        from piper import SynthesisConfig
        cfg = SynthesisConfig(speaker_id=int(rng.integers(self.speakers)),
                              length_scale=float(rng.uniform(0.8, 1.35)),
                              noise_scale=float(rng.uniform(0.4, 0.9)),
                              noise_w_scale=float(rng.uniform(0.5, 1.0)))
        pcm = b"".join(c.audio_int16_bytes for c in self.voice.synthesize(text, syn_config=cfg))
        return trim(resample(np.frombuffer(pcm, dtype=np.int16).astype(np.float32), self.sr))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def mix(speech: np.ndarray, noise: np.ndarray, rng: np.random.Generator, end_s: tuple[float, float],
        noise_gain: tuple[float, float] = (0.5, 1.6)) -> np.ndarray:
    """2 s window: `speech` ends at a random point in end_s, at a speech level like the real mic
    (RMS 300-2000, and at least 1.5x the noise: at the mic words were 2-10x the fan), over a
    random stretch of the recorded room noise at a random level. (v1 let the noise drown some
    examples, and the model learned "fan noise" as part of the word.)"""
    clip = np.zeros(WIN, dtype=np.float32)
    i = int(rng.integers(0, len(noise) - WIN))
    bg = noise[i:i + WIN] * rng.uniform(*noise_gain)
    end = int(rng.uniform(*end_s) * SR)
    s = speech[-end:] if len(speech) > end else speech
    if rms(s) > 0:
        s = s * (max(rng.uniform(300, 2000), 1.5 * rms(bg)) / rms(s))
    clip[end - len(s):end] = s
    clip += bg
    return np.clip(clip, -32768, 32767).astype(np.int16)


def features(clips: list[np.ndarray], feats) -> np.ndarray:
    out = []
    for i in range(0, len(clips), 64):
        out.append(feats.embed_clips(np.stack(clips[i:i + 64]), batch_size=64))
    return np.concatenate(out).astype(np.float32)


def make_clips(texts: list[str], n: int, synth: Synth, noise, rng, end_s, label: str) -> list[np.ndarray]:
    t, clips = time.perf_counter(), []
    for k in range(n):
        clips.append(mix(synth.say(texts[k % len(texts)], rng), noise, rng, end_s))
        if (k + 1) % 500 == 0:
            print(f"  {label}: {k + 1}/{n} ({time.perf_counter() - t:.0f} s)", flush=True)
    return clips


def export_onnx(mlp, path: Path) -> None:
    """MLP -> ONNX graph: (N,16,96) -> flatten -> [Gemm, Relu]* -> Gemm -> Sigmoid -> (N,1)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    nodes, inits = [helper.make_node("Flatten", ["input"], ["h0"], axis=1)], []
    h = "h0"
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        inits += [numpy_helper.from_array(W.astype(np.float32), f"W{i}"),
                  numpy_helper.from_array(b.astype(np.float32), f"b{i}")]
        nodes.append(helper.make_node("Gemm", [h, f"W{i}", f"b{i}"], [f"z{i}"]))
        last = i == len(mlp.coefs_) - 1
        nodes.append(helper.make_node("Sigmoid" if last else "Relu", [f"z{i}"], ["output" if last else f"h{i + 1}"]))
        h = f"h{i + 1}"
    graph = helper.make_graph(nodes, "wakeword",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 16, 96])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 1])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # readable by older onnxruntime builds too
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def stream_triggers(model_path: Path, audio: np.ndarray, threshold: float = 0.5) -> list[float]:
    """Run the real openWakeWord detector over a long recording; times (s) it would fire,
    with 2 s between triggers like the assistant (it beeps and listens after each)."""
    from openwakeword.model import Model
    m = Model(wakeword_model_paths=[str(model_path)])
    times, last = [], -9.0
    for i in range(0, len(audio) - 1280, 1280):
        score = max(m.predict(audio[i:i + 1280]).values())
        t = i / SR
        if score >= threshold and t - last > 2.0:
            times.append(round(t, 2))
            last = t
    return times



def real_word_clips(real: Path, word: str) -> list[np.ndarray]:
    """One clip per heard cue of the guided session (record_session.py): from the first to the last
    speech inside the cue's window, plus 100 ms, so words split by a pause stay whole."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from assistant.voice.audio import SileroVAD
    import wave
    with wave.open(str(real / f"word_{word}.wav")) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    meta = json.loads((real / f"word_{word}.json").read_text())
    vad = SileroVAD(0.5)
    raw = x.tobytes()
    probs = np.array([vad.speech_probability(raw[i:i + 960]) for i in range(0, len(raw) - 959, 960)])
    out = []
    for c in meta["cues"]:
        idx = np.flatnonzero(probs[c["start"]:c["end"]] >= 0.5)
        if not c["heard"] or not len(idx):
            continue
        a1 = (c["start"] + idx[-1] + 1) * 480 + 1600
        a0 = max((c["start"] + idx[0]) * 480 - 1600, a1 - int(1.5 * SR))  # one word: keep the last 1.5 s
        out.append(x[max(0, a0):a1].astype(np.float32))
    return out


def vary(seg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Speed/pitch change of +-10% (resampling), for more variety from a few real examples."""
    n = max(1, int(len(seg) / rng.uniform(0.9, 1.1)))
    return np.interp(np.linspace(0, len(seg) - 1, n), np.arange(len(seg)), seg).astype(np.float32)


def talk_windows(audio: np.ndarray, step_s: float = 0.25) -> list[np.ndarray]:
    return [audio[i:i + WIN].astype(np.int16) for i in range(0, len(audio) - WIN, int(step_s * SR))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("word", choices=sorted(SPELLINGS))
    ap.add_argument("--positives", type=int, default=2000)
    ap.add_argument("--sentences", type=int, default=2500, help="ordinary-speech negatives (cached)")
    ap.add_argument("--work", type=Path, default=Path.home() / "wake-train")
    ap.add_argument("--voice", type=Path, default=None)
    ap.add_argument("--real", type=Path, default=Path.home() / "wake-train" / "real2",
                    help="guided-session recordings (record_session.py); skipped if missing")
    a = ap.parse_args()
    rng = np.random.default_rng(sum(map(ord, a.word)))  # fixed per word: runs can be repeated
    t0 = time.perf_counter()

    import wave
    with wave.open(str(a.work / "room_noise.wav")) as w:
        noise = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    synth = Synth(a.voice or a.work / "voices" / "en_US-libritts_r-medium.onnx")
    from openwakeword.utils import AudioFeatures
    feats = AudioFeatures()
    print(f"{a.word}: voice with {synth.speakers} speakers, {len(noise) / SR:.0f} s of room noise", flush=True)

    cache = a.work / "cache" / f"common_negatives_{a.sentences}.npz"
    if cache.exists():
        common = np.load(cache)["x"]
        print(f"common negatives from cache: {len(common)}", flush=True)
    else:
        crng = np.random.default_rng(7)
        sent = make_clips(SENTENCES, a.sentences, synth, noise, crng, (0.3, 2.0), "sentences")
        quiet = [mix(np.zeros(1, np.float32), noise, crng, (1.0, 1.0)) for _ in range(500)]
        common = features(sent + quiet, feats)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, x=common)
        print(f"common negatives: {len(common)} ({time.perf_counter() - t0:.0f} s)", flush=True)

    others = [s for w, sp in SPELLINGS.items() if w != a.word for s in sp[:2]]
    own = {s.lower().strip("!") for s in SPELLINGS[a.word]}
    near_words = [w for w in CONFUSABLE[a.word] + others if w.lower() not in own]  # never both labels
    near = make_clips(near_words, 1200, synth, noise, rng, (1.6, 1.95), "similar words")
    quiet = [mix(np.zeros(1, np.float32), noise, rng, (1.0, 1.0), (0.3, 2.5)) for _ in range(1500)]
    # Running speech: 2 s slices from the middle of continuous talk. v2's sentence negatives all
    # contained a sentence end, so Boafo fired 18 times in 4 min of continuous speech.
    pool = [synth.say(SENTENCES[k % len(SENTENCES)], rng) for k in range(240)]
    print(f"  running-speech pool: {len(pool)} sentences", flush=True)
    running = []
    for _ in range(1500):
        talk = np.concatenate([np.concatenate([pool[j], np.zeros(int(rng.uniform(0.05, 0.4) * SR), np.float32)])
                               for j in rng.integers(0, len(pool), 4)])
        start = int(rng.integers(0, max(1, len(talk) - WIN)))
        running.append(mix(talk[start:start + WIN], noise, rng, (2.0, 2.0)))
    pos = make_clips(SPELLINGS[a.word], a.positives, synth, noise, rng, (1.6, 1.95), "positives")
    X_near, X_quiet, X_pos = features(near, feats), features(quiet + running, feats), features(pos, feats)

    # Real recordings from the room (guided session): the user's own examples of the word,
    # ordinary talk, and the user saying the other words. Synthetic-only Boafo fired every few
    # seconds on real speech (2026-11-20). 5 real examples and the last minute of talk are held out.
    real = {}
    if (a.real / f"word_{a.word}.json").exists():
        import wave
        mine = real_word_clips(a.real, a.word)
        test_mine, train_mine = mine[-5:], mine[:-5]
        aug = lambda clips, n: [mix(vary(c, rng), noise, rng, (1.6, 1.95)) for c in clips for _ in range(n)]  # noqa: E731
        with wave.open(str(a.real / "talk.wav")) as w:
            talk_audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        cut = len(talk_audio) - 60 * SR
        other_real = [c for w_ in SPELLINGS if w_ != a.word and (a.real / f"word_{w_}.json").exists()
                      for c in real_word_clips(a.real, w_)]
        real = {"pos": features(aug(train_mine, 40), feats),
                "pos_test": features(aug(test_mine, 40), feats),
                "talk": features(talk_windows(talk_audio[:cut]), feats),
                "other_words": features(aug(other_real, 10), feats),
                "talk_test_audio": talk_audio[cut:]}
        print(f"  real: {len(mine)} own examples ({len(train_mine)} train / {len(test_mine)} test), "
              f"{len(real['talk'])} talk windows, {len(other_real)} clips of the other words", flush=True)
    print(f"features done ({time.perf_counter() - t0:.0f} s)", flush=True)

    parts = [(X_pos, 1, 0), (X_near, 0, 1), (common, 0, 2), (X_quiet, 0, 3)]
    if real:
        parts += [(real["pos"], 1, 4), (real["talk"], 0, 5), (real["other_words"], 0, 6)]
    X = np.concatenate([x for x, _, _ in parts]).reshape(-1, 16 * 96)
    y = np.concatenate([np.full(len(x), lab, dtype=float) for x, lab, _ in parts])
    group = np.concatenate([np.full(len(x), g) for x, _, g in parts])
    idx = rng.permutation(len(X))
    test = idx[: len(idx) // 5]
    train = idx[len(idx) // 5:]

    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0)
    mlp.fit(X[train], y[train])
    p = mlp.predict_proba(X[test])[:, 1]
    yt, gt = y[test], group[test]
    report = {"word": a.word, "positives": int(len(X_pos)), "similar_words": int(len(X_near)),
              "common_negatives": int(len(common)), "noise_negatives": int(len(X_quiet)),
              "held_out": int(len(test))}
    for thr in (0.5, 0.7, 0.9):
        report[f"recall@{thr}"] = round(float((p[yt == 1] >= thr).mean()), 4)
        report[f"false_accept_similar@{thr}"] = round(float((p[gt == 1] >= thr).mean()), 4)
        report[f"false_accept_other@{thr}"] = round(float((p[gt == 2] >= thr).mean()), 4)
        report[f"false_accept_noise_or_running_speech@{thr}"] = round(float((p[gt == 3] >= thr).mean()), 4)
        if real:
            pr = mlp.predict_proba(real["pos_test"].reshape(-1, 16 * 96))[:, 1]
            report[f"REAL_recall_held_out@{thr}"] = round(float((pr >= thr).mean()), 4)
            report[f"false_accept_real_talk@{thr}"] = round(float((p[gt == 5] >= thr).mean()), 4)
            report[f"false_accept_real_other_words@{thr}"] = round(float((p[gt == 6] >= thr).mean()), 4)
    print(json.dumps(report, indent=1), flush=True)

    out = ROOT / "models" / "wake" / f"{a.word}.onnx"
    export_onnx(mlp, out)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_p = sess.run(None, {"input": X[test[:50]].reshape(-1, 16, 96)})[0][:, 0]
    report["onnx_matches_sklearn"] = bool(np.allclose(onnx_p, p[:50], atol=1e-4))

    # The real test of false alarms: the detector running over 90 s of room noise plus a stream of
    # ordinary sentences it never trained on, the way it runs in the assistant.
    srng = np.random.default_rng(99)
    talk = [synth.say(s, srng) * (srng.uniform(300, 2000) / 3000) for s in TEST_SENTENCES * 3]
    gap = np.zeros(int(0.4 * SR), np.float32)
    speech = np.concatenate([x for s in talk for x in (s, gap)])
    bg = np.resize(noise, len(speech))
    stream = np.clip(np.concatenate([noise, speech + bg]), -32768, 32767).astype(np.int16)
    fires = stream_triggers(out, stream)
    report["stream_minutes"] = round(len(stream) / SR / 60, 1)
    report["stream_false_triggers"] = len(fires)
    # The assistant only accepts a wake word while Silero hears speech: count what would remain.
    sys_path = str(ROOT / "src")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from assistant.voice.audio import SileroVAD
    vad = SileroVAD(0.5)
    sp = np.array([vad.is_speech(stream[i:i + 480].tobytes()) for i in range(0, len(stream) - 480, 480)])
    def speech_near(t: float) -> bool:
        k = int(t * SR / 480)
        return bool(sp[max(0, k - 33):k + 1].any())   # speech within the last second
    report["stream_false_triggers_with_silero"] = sum(speech_near(t) for t in fires)
    if real:  # the minute of the user's real talk that training never saw
        real_fires = stream_triggers(out, real["talk_test_audio"].astype(np.int16))
        report["REAL_talk_minute_false_triggers"] = len(real_fires)
    report["seconds_total"] = round(time.perf_counter() - t0)
    (a.work / f"{a.word}_report.json").write_text(json.dumps(report, indent=1))
    print(f"saved {out}; onnx matches: {report['onnx_matches_sklearn']}; false triggers in "
          f"{report['stream_minutes']} min of noise + other speech: {len(fires)} {fires[:10]} "
          f"({report['stream_false_triggers_with_silero']} with the Silero check); "
          f"real talk minute: {report.get('REAL_talk_minute_false_triggers', 'n/a')}", flush=True)


if __name__ == "__main__":
    main()
