"""GPU engine process: continuous batching behind a Unix-domain-socket protocol.

The HTTP layer (the Go gateway in `gateway/`) lives in another process, so this process does nothing
but drive the GPU: it reads requests from one socket and writes every stream's audio for a step in a
single `sendall`.

Frames (little endian): header = type u8 | request id u32 | payload length u32, then the payload.
    1 REQ     gateway -> engine   JSON {"input", "voice", "language"}
    2 AUDIO   engine  -> gateway  raw PCM s16le, 24 kHz mono
    3 END     engine  -> gateway  (empty)
    4 ERR     engine  -> gateway  UTF-8 message
    5 CANCEL  gateway -> engine   (empty) client went away
    6 READY   engine  -> gateway  (empty) sent once the engine is warm

    python -m uqt.engine_proc --sock /tmp/uqt.sock --max-slots 512 --max-len 512
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import struct
import threading
import time

import numpy as np

HDR = struct.Struct("<BII")
REQ, AUDIO, END, ERR, CANCEL, READY = 1, 2, 3, 4, 5, 6


def _pcm(w):
    return (np.clip(w, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


HOPS = {"recv_to_admit": [], "admit_to_first": []}


def _emit(be, outs, cancelled, frames):
    """Append AUDIO/END frames for finished decodes; release flushed slots."""
    for s, b, fin in outs:
        meta = be.info[s]["meta"]
        rid = meta["id"]
        if b and "first" not in meta and "recv" in meta:
            meta["first"] = time.perf_counter()
            HOPS["recv_to_admit"].append(meta["admit"] - meta["recv"])
            HOPS["admit_to_first"].append(meta["first"] - meta["admit"])
        if b and rid not in cancelled:
            frames.append(HDR.pack(AUDIO, rid, len(b)))
            frames.append(b)
        if fin:
            frames.append(HDR.pack(END, rid, 0))
            cancelled.discard(rid)
            be.release(s)


def _reader(conn, q, cancelled):
    buf = bytearray()
    while True:
        data = conn.recv(1 << 20)
        if not data:
            os._exit(0)  # gateway gone
        buf += data
        while len(buf) >= HDR.size:
            t, rid, n = HDR.unpack_from(buf, 0)
            if len(buf) < HDR.size + n:
                break
            payload = bytes(buf[HDR.size:HDR.size + n])
            del buf[:HDR.size + n]
            if t == REQ:
                req = json.loads(payload)
                req["_recv"] = time.perf_counter()
                q.put((rid, req))
            elif t == CANCEL:
                cancelled.add(rid)


def selfwarm(eng, be, waves=(1, 16, 64, 256, 512), K=2, max_frames=120):
    """Run synthetic streams through the real admission / generation / decode path at rising
    concurrency so every lazily compiled kernel variant exists before the first client arrives."""
    import torch

    texts = ["Hello there.", "This is a warm-up sentence for the speech engine.",
             "Every lazily compiled kernel should exist before the first real request arrives, so we run a few waves.",
             "Short.", "The quick brown fox jumps over the lazy dog, again and again, until the engine is warm."]
    t0 = time.perf_counter()
    with torch.inference_mode():
        for n in waves:
            n = min(n, be.S)
            pend = [(i, texts[i % len(texts)]) for i in range(n)]
            handle = None
            while pend or handle is not None or any(it is not None for it in be.info):
                batch, taken = [], set()
                while pend and len(batch) < 16:
                    slot = next((i for i, it in enumerate(be.info) if it is None and i not in taken), None)
                    if slot is None:
                        break
                    taken.add(slot)
                    _, tx = pend.pop()
                    batch.append((slot, eng.build_prompt("custom", text=tx, language="English", speaker="Ryan"),
                                  max_frames, {"id": 0}))
                if batch:
                    be.admit_many(batch)
                if handle is not None or be.active():
                    be.collect(handle) if handle is not None else be.generate(K)
                    handle = be.launch(K) if be.active() else None
                for s, _w, fin in be.decode_due():
                    if fin:
                        be.release(s)
    torch.cuda.synchronize()
    print(f"[uqt.engine_proc] self-warm {waves} in {time.perf_counter() - t0:.0f}s", flush=True)


def serve(eng, be, conn, K=2, admit_per_iter=16, max_frames=400, fast_lane=True):
    import torch

    q: queue.Queue = queue.Queue()
    cancelled: set = set()
    threading.Thread(target=_reader, args=(conn, q, cancelled), daemon=True).start()
    conn.sendall(HDR.pack(READY, 0, 0))
    handle = None
    pending_dec = None
    stats = {"iters": 0, "admit": 0.0, "gen": 0.0, "dec": 0.0, "send": 0.0, "dec_calls": 0, "admitted": 0,
             "active": 0, "t": time.perf_counter()}
    with torch.inference_mode():
        while True:
            if handle is None and pending_dec is None and not be.active() and all(it is None for it in be.info):
                q.put(q.get())  # idle: block until a request arrives
            t_a = time.perf_counter()
            # admission
            batch, taken = [], set()
            while len(batch) < admit_per_iter:
                slot = next((i for i, it in enumerate(be.info) if it is None and i not in taken), None)
                if slot is None:
                    break
                try:
                    rid, req = q.get_nowait()
                except queue.Empty:
                    break
                if rid in cancelled:
                    continue
                try:
                    p = eng.build_prompt("custom", text=req["input"], language=req.get("language") or "English",
                                         speaker=req.get("voice") or "Ryan")
                except Exception as e:  # bad request: report and move on
                    msg = repr(e)[:500].encode()
                    conn.sendall(HDR.pack(ERR, rid, len(msg)) + msg)
                    continue
                taken.add(slot)
                batch.append((slot, p, max_frames, {"id": rid, "recv": req["_recv"], "admit": time.perf_counter()}))
            # the newest request takes the fast lane (run after the batched step is launched, below)
            # fast lane only while the GPU has slack: under load its concurrent work slows the batched step
            fast_max = int(os.environ.get("UQT_FAST_MAX", "8"))
            fast = batch.pop() if (fast_lane and batch and len(be.active()) <= fast_max) else None
            if batch:
                be.admit_many(batch)
                stats["admitted"] += len(batch)
            t_g = time.perf_counter()
            # cancellations: stop generating, drop pending audio
            if cancelled:
                for s, it in enumerate(be.info):
                    if it is not None and it["meta"]["id"] in cancelled:
                        it["done"] = True
                        it["valid"] = it["dec"]
            # generation (collect step i, launch step i+1 before decoding)
            if handle is not None or be.active():
                be.collect(handle) if handle is not None else be.generate(K)
                handle = be.launch(K) if be.active() else None
            frames = []
            if fast is not None:
                slot, p, mf, meta = fast
                b, fin = be.admit_fast(slot, p, mf, meta)
                rid = meta["id"]
                meta["first"] = time.perf_counter()
                HOPS["recv_to_admit"].append(meta["admit"] - meta["recv"])
                HOPS["admit_to_first"].append(meta["first"] - meta["admit"])
                if b:
                    frames += [HDR.pack(AUDIO, rid, len(b)), b]
                if fin:
                    frames.append(HDR.pack(END, rid, 0))
                stats["admitted"] += 1
            # decode: launch this step's decodes asynchronously, deliver the previous step's results.
            # At light load deliver immediately (lower latency); under load overlap decode with generation.
            t_d = time.perf_counter()
            n0 = getattr(be, "dec_calls", 0)
            firsts, launched = be.decode_launch(split_first=True)
            if firsts:  # a stream's first chunk is never held back
                _emit(be, be.decode_finish(firsts), cancelled, frames)
            if pending_dec is not None:
                _emit(be, be.decode_finish(pending_dec), cancelled, frames)
                pending_dec = None
            if len(be.active()) <= 16:
                _emit(be, be.decode_finish(launched), cancelled, frames)
            else:
                pending_dec = launched
            t_s = time.perf_counter()
            if frames:
                conn.sendall(b"".join(frames))
            t_e = time.perf_counter()
            if os.environ.get("UQT_STATS"):
                stats["iters"] += 1
                stats["admit"] += t_g - t_a
                stats["gen"] += t_d - t_g
                stats["dec"] += t_s - t_d
                stats["send"] += t_e - t_s
                stats["dec_calls"] += getattr(be, "dec_calls", 0) - n0
                stats["active"] += len(be.active())
                stats["occupied"] = stats.get("occupied", 0) + sum(1 for it in be.info if it is not None)
                stats["queued"] = stats.get("queued", 0) + q.qsize()
                if t_e - stats["t"] > 2.0:
                    it_ = max(stats["iters"], 1)
                    hops = {k: (round(1000 * float(np.percentile(v, 90)), 1) if v else None) for k, v in HOPS.items()}
                    for v in HOPS.values():
                        v.clear()
                    print("[uqt.hops p90 ms]", json.dumps(hops), flush=True)
                    print("[uqt.stats]", json.dumps({k: (round(v / it_ * 1000, 2) if k in ("admit", "gen", "dec", "send") else
                                                         (round(v / it_, 1) if k in ("active", "dec_calls", "occupied", "queued") else v))
                                                    for k, v in stats.items() if k != "t"}), flush=True)
                    stats = {k: 0 if k != "t" else t_e for k in stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/tmp/uqt.sock")
    ap.add_argument("--variant", default="CustomVoice")
    ap.add_argument("--max-slots", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    import torch
    from qwen_tts import Qwen3TTSModel

    from uqt.batch_engine import BatchEngine
    from uqt.engine import Engine

    t0 = time.time()
    torch.set_grad_enabled(False)
    tts = Qwen3TTSModel.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-1.7B-{args.variant}", device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    with torch.inference_mode():
        eng = Engine(tts, max_len=2048, streaming_codec=False, autotune=False)
        be = BatchEngine(eng, max_slots=args.max_slots, max_len=args.max_len)
        be.warmup(Ks=(2,))
        wp = eng.build_prompt("custom", text="Warming up the fast lane for new requests.", language="English",
                              speaker="Ryan")
        eng.warmup(wp, Ks=(2,), max_bucket=min(1024, args.max_len), prefill_max=512)
        for i, tx in enumerate(["Hi.", "A somewhat longer warm-up sentence for the fast lane path.",
                                "An even longer warm-up sentence, so that a larger prefill bucket is exercised "
                                "by the fast lane before any real request arrives at the server."]):
            p = eng.build_prompt("custom", text=tx, language="English", speaker="Ryan")
            be.admit_fast(i, p, 50, {"id": 0})
            be.release(i)
        torch.cuda.synchronize()
    selfwarm(eng, be)
    if os.path.exists(args.sock):
        os.unlink(args.sock)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.sock)
    srv.listen(1)
    print(f"[uqt.engine_proc] warm in {time.time() - t0:.0f}s, waiting for gateway on {args.sock}", flush=True)
    conn, _ = srv.accept()
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
    serve(eng, be, conn)


if __name__ == "__main__":
    main()
