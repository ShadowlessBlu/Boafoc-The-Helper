"""Voice pipeline: microphone -> wake word -> VAD -> STT -> Brain -> TTS -> speaker.

Heavy engines (openWakeWord, Vosk, whisper.cpp, Piper) are optional and only
imported/launched when used, so the rest of the assistant runs without them.
"""
