"""Small helpers shared by benchmark scripts (kept dependency-free)."""
import hashlib

TIMING_TEXTS = [
    "Hello, how can I help you today?",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Please confirm your appointment for Thursday at three thirty in the afternoon, or call us to reschedule.",
    "Speech synthesis has improved dramatically, and modern systems can now produce natural, expressive voices in real time on a single GPU.",
    "When the storm finally passed, the villagers walked down to the harbor, counted the boats that had survived the night, and began repairing the ones that had not.",
]


def seed_for(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def first_voiced_sample(wav, sr, thresh_db=-40.0, win_ms=10):
    import numpy as np

    win = max(1, int(sr * win_ms / 1000))
    if not len(wav) or float(np.max(np.abs(wav))) == 0.0:
        return None
    thr = 10 ** (thresh_db / 20)
    for i in range(0, len(wav) - win + 1, win):
        if float(np.sqrt(np.mean(wav[i:i + win] ** 2))) >= thr:
            return i
    return None
