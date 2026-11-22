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
    """Index of the first 10 ms window with RMS above -40 dBFS (vectorized: no Python loop per window,
    so measuring one stream does not stall the event loop that receives all the others)."""
    win = SR // 100
    n = len(pcm) // win
    if n == 0:
        return None
    rms = np.sqrt(np.mean(pcm[: n * win].reshape(n, win) ** 2, axis=1))
    hit = np.flatnonzero(rms >= THRESH)
    return int(hit[0]) * win if len(hit) else None


async def _one(client, url, payload, extra_headers=None):
    import httpx

    t0 = time.perf_counter()
    chunks = []  # (t, bytes)
    async with client.stream("POST", url, json=payload, headers=extra_headers or {},
                             timeout=httpx.Timeout(600.0)) as r:
        t_hdr = time.perf_counter()
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
    rec = {"t0": t0, "n_bytes": len(raw), "audio_s": dur, "n_chunks": len(chunks), "hdr_ms": (t_hdr - t0) * 1000}
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
        "hdr_ms_p50": _pct(g("hdr_ms"), 50), "hdr_ms_p90": _pct(g("hdr_ms"), 90),
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


def _load_shard(args):
    """One load-generator process: n clients (global indices offset..offset+n) of a C-client run."""
    import httpx

    url, payloads, offset, n, C, reqs_per_client, ramp_s, t_start, headers = args

    async def run():
        limits = httpx.Limits(max_connections=n + 8, max_keepalive_connections=n + 8)
        async with httpx.AsyncClient(limits=limits) as client:
            recs = []

            async def worker(ci):
                g = offset + ci
                await asyncio.sleep(max(0.0, t_start - time.time()) + g * ramp_s / C)
                for j in range(reqs_per_client):
                    p = payloads[(g * reqs_per_client + j) % len(payloads)]
                    try:
                        rec, _pcm = await _one(client, url, p, headers)
                    except Exception as e:  # count failures, keep going
                        rec = {"error": repr(e)[:200], "audio_s": 0}
                    rec.pop("t0", None)
                    rec["end_wall"] = time.time()
                    recs.append(rec)

            await asyncio.gather(*[worker(i) for i in range(n)])
            return recs

    return asyncio.run(run())


def load(url, make_payload, texts, concs=(1, 8, 32, 64, 128, 256), reqs_per_client=4, ramp_s=0.5,
         headers=None, max_connections=1024, clients_per_proc=32, max_procs=16):
    """Closed-loop back-to-back clients at each concurrency; returns per-concurrency summaries.

    A single asyncio/httpx process saturates around 200 audio-seconds per second of streamed PCM, so
    the clients are spread over several processes (about `clients_per_proc` each, at most `max_procs`)
    that start together and report per-request records."""
    import multiprocessing as mp

    here = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = here + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    payloads = [make_payload(t) for t in texts]
    ctx = mp.get_context("spawn")
    out = {}
    for C in concs:
        procs = max(1, min(max_procs, -(-C // clients_per_proc)))
        sizes = [C // procs + (1 if i < C % procs else 0) for i in range(procs)]
        offs = [sum(sizes[:i]) for i in range(procs)]
        t_start = time.time() + 2.0 + 0.2 * procs  # let every process import and connect first
        jobs = [(url, payloads, offs[i], sizes[i], C, reqs_per_client, ramp_s, t_start, headers) for i in range(procs)]
        if procs == 1:
            parts = [_load_shard(jobs[0])]
        else:
            with ctx.Pool(procs) as pool:
                parts = pool.map(_load_shard, jobs)
        recs = [r for part in parts for r in part]
        wall = max(r["end_wall"] for r in recs) - t_start
        s = _summ(recs)
        s.update({"C": C, "procs": procs, "wall_s": wall, "audio_s": sum(r.get("audio_s", 0) for r in recs),
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
