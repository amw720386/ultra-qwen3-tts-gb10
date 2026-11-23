"""WER of audio that came through the running server under load (unlike acc_batch, this covers the serving
loop: chunk order, trimming, delivery). Also reports clips whose chunks fit the reference length badly.

    python -m bench.gb10.server_wer --conc 128 --out /root/results/swer-c128
"""
import argparse
import asyncio
import json
import os
import time

import numpy as np
import soundfile as sf


async def collect(url, items, conc):
    import httpx

    out, sem = {}, asyncio.Semaphore(conc)
    limits = httpx.Limits(max_connections=conc + 8)
    async with httpx.AsyncClient(limits=limits) as c:
        async def one(it):
            async with sem:
                buf = []
                async with c.stream("POST", url, json={"input": it["target_text"], "voice": "Ryan", "language": "English",
                                                       "response_format": "pcm", "stream": True},
                                    timeout=httpx.Timeout(600.0)) as r:
                    if r.status_code != 200:
                        raise RuntimeError(f"HTTP {r.status_code}")
                    async for b in r.aiter_bytes():
                        buf.append(b)
                raw = b"".join(buf)
                out[it["id"]] = raw[: len(raw) - len(raw) % 2]
        await asyncio.gather(*[one(i) for i in items])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091/v1/audio/speech")
    ap.add_argument("--conc", type=int, default=128)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--manifest", default="/root/manifest.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    man = json.load(open(args.manifest))[: args.n]
    t0 = time.time()
    pcm = asyncio.run(collect(args.url, man, args.conc))
    os.makedirs(args.out, exist_ok=True)
    for k, b in pcm.items():
        sf.write(f"{args.out}/{k}.wav", np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0, 24000)
    print(f"{len(pcm)} clips in {time.time() - t0:.0f}s", flush=True)
    from uqt.evaluation import run_eval

    r = run_eval(args.out, man, with_sim=False)
    bad = [x for x in r["rows"] if x["errors"] / max(x["n_ref"], 1) > 0.5]
    ok = [x for x in r["rows"] if x not in bad]
    summ = {"n": r["n_eval"], "wer": r["wer_corpus"], "catastrophic": len(bad),
            "wer_excl": sum(x["errors"] for x in ok) / max(sum(x["n_ref"] for x in ok), 1), "conc": args.conc}
    json.dump({"summary": summ, **r}, open(f"{args.out}/eval.json", "w"), indent=1)
    print("server-wer", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
