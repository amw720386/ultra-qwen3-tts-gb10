"""Non-blocking copies must not change a sample: same seed, same admission order, compare every clip.

    python -m bench.gb10.async_equal --slots 32 --n 64
"""
import argparse
import glob
import hashlib
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--manifest", default="/root/manifest.json")
    args = ap.parse_args()
    from qwen_tts import Qwen3TTSModel

    from bench.gb10.acc_batch import generate
    from uqt import kernels
    from uqt.batch_engine import BatchEngine
    from uqt.engine import Engine

    man = json.load(open(args.manifest))[: args.n]
    torch.set_grad_enabled(False)
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    sums = {}
    with torch.inference_mode():
        eng = Engine(tts, max_len=2048, streaming_codec=False, autotune=False, weights="hyb")
        be = BatchEngine(eng, max_slots=args.slots, max_len=512)
        be.warmup(Ks=(2,))
        # "off2" repeats "off": how much two identical runs differ is the noise floor
        for name, on in (("warm", 0), ("off", 0), ("off2", 0), ("on", 1), ("on2", 1)):
            kernels.ASYNC_H2D[0] = on
            torch.manual_seed(7)
            d = f"/root/results/async-{name}"
            generate(eng, be, man, d)
            sums[name] = {f.split("/")[-1]: hashlib.md5(open(f, "rb").read()).hexdigest()
                          for f in glob.glob(f"{d}/*.wav")}
    ref = sums["off"]
    for name in ("off2", "on", "on2"):
        print(name, "identical clips vs off:", sum(sums[name].get(k) == v for k, v in ref.items()), "of", len(ref),
              flush=True)


if __name__ == "__main__":
    main()
