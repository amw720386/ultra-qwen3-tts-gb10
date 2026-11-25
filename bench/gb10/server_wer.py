"""WER of audio that came through the running server under load (unlike acc_batch, this covers the serving
loop: chunk order, trimming, delivery). Also reports clips whose chunks fit the reference length badly.

    python -m bench.gb10.server_wer --conc 128 --out /root/results/swer-c128

--gen-only and --score-only split the two halves, so audio from several servers can be scored together
afterwards. A request that fails is recorded (`failed` in gen.json), not retried.
"""
import argparse
import asyncio
import hashlib
import json
import os
import time

import numpy as np
import soundfile as sf


async def collect(url, items, conc, extra=None):
    import httpx

    out, failed, sem = {}, {}, asyncio.Semaphore(conc)
    limits = httpx.Limits(max_connections=conc + 8)
    async with httpx.AsyncClient(limits=limits) as c:
        async def one(it):
            async with sem:
                buf = []
                try:
                    async with c.stream("POST", url, json={"input": it["target_text"], "voice": "Ryan", "language": "English",
                                                           "response_format": "pcm", "stream": True, **(extra or {})},
                                        timeout=httpx.Timeout(600.0)) as r:
                        if r.status_code != 200:
                            failed[it["id"]] = f"HTTP {r.status_code}: {(await r.aread())[:200]!r}"
                            return
                        async for b in r.aiter_bytes():
                            buf.append(b)
                except httpx.HTTPError as e:
                    failed[it["id"]] = repr(e)
                    return
                raw = b"".join(buf)
                if len(raw) < 2:
                    failed[it["id"]] = "no audio"
                    return
                out[it["id"]] = raw[: len(raw) - len(raw) % 2]
        await asyncio.gather(*[one(i) for i in items])
    return out, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091/v1/audio/speech")
    ap.add_argument("--conc", type=int, default=128)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--manifest", default="/root/manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="{}", help="extra JSON fields for the request payload")
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    man = json.load(open(args.manifest))[: args.n]
    os.makedirs(args.out, exist_ok=True)
    if not args.score_only:
        t0 = time.time()
        pcm, failed = asyncio.run(collect(args.url, man, args.conc, json.loads(args.extra)))
        wall = time.time() - t0
        for k, b in pcm.items():
            sf.write(f"{args.out}/{k}.wav", np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0, 24000)
        json.dump({"conc": args.conc, "wall_s": wall, "failed": failed,
                   "audio_s": {k: len(b) / 48000 for k, b in pcm.items()},
                   "md5": {k: hashlib.md5(b).hexdigest() for k, b in pcm.items()}},
                  open(f"{args.out}/gen.json", "w"), indent=1)
        print(f"{len(pcm)} clips, {len(failed)} failed, in {wall:.0f}s", flush=True)
        if args.gen_only:
            return
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
