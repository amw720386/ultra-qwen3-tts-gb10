"""Shared head-to-head client: streams PCM from an HTTP TTS endpoint and measures every system the same way.

Every system under test runs as an HTTP server inside the same Modal container as this client
(localhost), returning raw 16-bit little-endian mono PCM at 24 kHz as a chunked/streamed response.

Metrics per request (all wall clock on the client):
  ttfa_ms        request sent -> first non-empty audio bytes received
  ttfa_voiced_ms request sent -> first received sample above -40 dBFS (10 ms RMS window)
  rtf            (last byte - first byte) / audio duration      (lower is faster)
  total_rtf      (last byte - request sent) / audio duration
  max_late_ms    worst chunk lateness vs real-time playback started at the first byte (> 0 = underrun)

Scenarios:
  latency(...)   sequential requests after warmup; saves wavs for WER
  load(...)      closed-loop back-to-back clients at each concurrency (0.5 s ramp)

Usage inside a container (after the server is up):
    from client import latency, load
    res = latency(url, make_payload, texts, out_dir="/data/gen/h2h-vllm")
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import numpy as np

SR = 24000
THRESH = 10 ** (-40 / 20)


def _voiced_offset(pcm: np.ndarray):
    win = SR // 100
    for i in range(0, len(pcm) - win + 1, win):
        if float(np.sqrt(np.mean(pcm[i:i + win] ** 2))) >= THRESH:
            return i
    return None


async def _one(client, url, payload, extra_headers=None):
    import httpx

    t0 = time.perf_counter()
    chunks = []  # (t, bytes)
    async with client.stream("POST", url, json=payload, headers=extra_headers or {},
                             timeout=httpx.Timeout(600.0)) as r:
        if r.status_code != 200:
            body = await r.aread()
            raise RuntimeError(f"HTTP {r.status_code}: {body[:300]!r}")
        async for b in r.aiter_bytes():
            if b:
                chunks.append((time.perf_counter(), b))
    t_end = time.perf_counter()
    raw = b"".join(c for _, c in chunks)
    raw = raw[: len(raw) - (len(raw) % 2)]
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    dur = len(pcm) / SR
    rec = {"t0": t0, "n_bytes": len(raw), "audio_s": dur, "n_chunks": len(chunks)}
    if not chunks or dur == 0:
        rec["error"] = "no audio"
        return rec, pcm
    t_first = chunks[0][0]
    rec["ttfa_ms"] = (t_first - t0) * 1000
    v = _voiced_offset(pcm)
    if v is not None:
        acc = 0
        for t, c in chunks:
            acc += len(c) // 2
            if acc > v:
                rec["ttfa_voiced_ms"] = (t - t0) * 1000
                break
    rec["rtf"] = (t_end - t_first) / dur
    rec["total_rtf"] = (t_end - t0) / dur
    acc, worst = 0.0, -1e9
    for t, c in chunks:
        worst = max(worst, t - (t_first + acc))
        acc += (len(c) // 2) / SR
    rec["max_late_ms"] = worst * 1000
    return rec, pcm


def _pct(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def _summ(recs):
    ok = [r for r in recs if "error" not in r]
    g = lambda k: [r[k] for r in ok if r.get(k) is not None]  # noqa: E731
    return {
        "n": len(recs), "n_ok": len(ok),
        "ttfa_ms_p50": _pct(g("ttfa_ms"), 50), "ttfa_ms_p90": _pct(g("ttfa_ms"), 90), "ttfa_ms_p99": _pct(g("ttfa_ms"), 99),
        "ttfa_voiced_ms_p50": _pct(g("ttfa_voiced_ms"), 50), "ttfa_voiced_ms_p90": _pct(g("ttfa_voiced_ms"), 90),
        "rtf_p50": _pct(g("rtf"), 50), "rtf_p90": _pct(g("rtf"), 90),
        "total_rtf_p50": _pct(g("total_rtf"), 50),
        "underrun_frac": sum(1 for r in ok if r.get("max_late_ms", 0) > 0) / max(len(ok), 1),
    }


def latency(url, make_payload, texts, ids=None, out_dir=None, warmup=3, headers=None):
    """Sequential requests. make_payload(text) -> json body. Saves <id>.wav per text when out_dir is set."""
    import httpx
    import soundfile as sf

    async def run():
        async with httpx.AsyncClient() as client:
            for t in texts[:warmup]:
                await _one(client, url, make_payload(t), headers)
            recs = []
            for i, t in enumerate(texts):
                rec, pcm = await _one(client, url, make_payload(t), headers)
                rec["text_idx"] = i
                recs.append(rec)
                if out_dir and len(pcm):
                    os.makedirs(out_dir, exist_ok=True)
                    sf.write(f"{out_dir}/{(ids or list(range(len(texts))))[i]}.wav", pcm, SR)
            return recs

    recs = asyncio.run(run())
    return {"summary": _summ(recs), "requests": [{k: v for k, v in r.items() if k != "t0"} for r in recs]}


def load(url, make_payload, texts, concs=(1, 8, 32, 64, 128, 256), reqs_per_client=4, ramp_s=0.5,
         headers=None, max_connections=1024):
    """Closed-loop back-to-back clients at each concurrency; returns per-concurrency summaries."""
    import httpx

    out = {}
    for C in concs:
        async def run():
            limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
            async with httpx.AsyncClient(limits=limits) as client:
                recs = []
                k = [0]

                async def worker(ci):
                    await asyncio.sleep(ci * ramp_s / C)
                    for _ in range(reqs_per_client):
                        t = texts[k[0] % len(texts)]
                        k[0] += 1
                        try:
                            rec, pcm = await _one(client, url, make_payload(t), headers)
                        except Exception as e:  # count failures, keep going
                            rec = {"error": repr(e)[:200], "t0": time.perf_counter(), "audio_s": 0}
                        rec["end"] = time.perf_counter()
                        recs.append(rec)

                t0 = time.perf_counter()
                await asyncio.gather(*[worker(i) for i in range(C)])
                wall = time.perf_counter() - t0
                return recs, wall

        recs, wall = asyncio.run(run())
        s = _summ(recs)
        s.update({"C": C, "wall_s": wall, "audio_s": sum(r.get("audio_s", 0) for r in recs),
                  "throughput_audio_s_per_s": sum(r.get("audio_s", 0) for r in recs) / wall,
                  "errors": [r["error"] for r in recs if "error" in r][:3]})
        print("h2h-load", json.dumps(s), flush=True)
        out[C] = s
    return out


def wait_ready(url, timeout_s=1800, probe=None):
    """Poll `url` (GET) until it answers 200, or run `probe()` until it returns True."""
    import httpx

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if probe is not None:
                if probe():
                    return True
            elif httpx.get(url, timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"server not ready after {timeout_s}s: {url}")
