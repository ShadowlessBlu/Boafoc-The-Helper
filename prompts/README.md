# Prompts

- `standard_v1.txt`: fixed prompts for speed benchmarks (`bench/bench_ollama.py`). Never edit after results exist; add `standard_v2.txt` instead.
- The assistant's system prompt lives in `src/assistant/brain.py` (`SYSTEM_TEMPLATE`), so the assistant and `experiments/score.py` always use the same one. Changing it is a new experiment: note the change in the Experiment Log.
