"""OpenAI-style streaming speech server for the uqt engines.

    python -m uqt.server --mode batch  --variant CustomVoice --port 8091
    python -m uqt.server --mode single --variant CustomVoice --port 8091

POST /v1/audio/speech  {"input": str, "voice": str, "language": "English"}  ->  chunked raw PCM
(16-bit little-endian, mono, 24 kHz). GET /health returns 200 once warm.

* batch:  continuous batching (uqt.batch_engine.BatchEngine) on a dedicated GPU thread.
* single: the single-stream engine (uqt.engine.Engine.stream); requests are served one at a time.
"""
from __future__ import annotations

import argparse
import asyncio
import queue
import threading
import time

import numpy as np

STATE = {"ready": False}


def _pcm(w):
    return (np.clip(w, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class _Worker:
    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.t = threading.Thread(target=self._run, daemon=True)

    def submit(self, req):
        self.q.put(req)


class BatchWorker(_Worker):
    def __init__(self, eng, be, K=2, admit_per_iter=16, max_frames=400):
        super().__init__()
        self.eng, self.be, self.K, self.admit, self.max_frames = eng, be, K, admit_per_iter, max_frames

    def _run(self):
        import torch

        be, eng, K = self.be, self.eng, self.K
        handle = None
        with torch.inference_mode():
            while True:
                if not be.active() and handle is None:
                    req = self.q.get()  # idle: block for the next request
                    self.q.put(req)
                batch, taken = [], set()
                while len(batch) < self.admit:
                    slot = next((i for i, it in enumerate(be.info) if it is None and i not in taken), None)
                    if slot is None:
                        break
                    try:
                        req = self.q.get_nowait()
                    except queue.Empty:
                        break
                    taken.add(slot)
                    p = eng.build_prompt("custom", text=req["text"], language=req["language"], speaker=req["voice"])
                    batch.append((slot, p, self.max_frames, req))
                if batch:
                    be.admit_many(batch)
                slots, chunk, valid = be.collect(handle) if handle is not None else be.generate(K)
                handle = be.launch(K) if be.active() else None
                if not slots:
                    continue
                wavs = be.decode(slots, chunk, valid)
                for s, w in zip(slots, wavs):
                    it = be.info[s]
                    req = it["meta"]
                    if len(w):
                        req["loop"].call_soon_threadsafe(req["out"].put_nowait, _pcm(w))
                    if it["done"]:
                        req["loop"].call_soon_threadsafe(req["out"].put_nowait, None)
                        be.release(s)


class SingleWorker(_Worker):
    def __init__(self, eng, schedule=(1, 2, 4, 8)):
        super().__init__()
        self.eng, self.schedule = eng, schedule

    def _run(self):
        import torch

        with torch.inference_mode():
            while True:
                req = self.q.get()
                p = self.eng.build_prompt("custom", text=req["text"], language=req["language"], speaker=req["voice"])
                for w in self.eng.stream(p, schedule=self.schedule):
                    req["loop"].call_soon_threadsafe(req["out"].put_nowait, _pcm(w))
                req["loop"].call_soon_threadsafe(req["out"].put_nowait, None)


def build_app(worker):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
    from starlette.routing import Route

    async def speech(request):
        body = await request.json()
        out: asyncio.Queue = asyncio.Queue()
        worker.submit({"text": body["input"], "voice": body.get("voice", "Ryan"),
                       "language": body.get("language", "English"), "loop": asyncio.get_running_loop(), "out": out})

        async def gen():
            while True:
                b = await out.get()
                if b is None:
                    break
                yield b

        return StreamingResponse(gen(), media_type="audio/pcm")

    async def health(request):
        return PlainTextResponse("ok") if STATE["ready"] else JSONResponse({"ready": False}, status_code=503)

    return Starlette(routes=[Route("/v1/audio/speech", speech, methods=["POST"]), Route("/health", health)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["batch", "single"], default="batch")
    ap.add_argument("--variant", default="CustomVoice")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--max-slots", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    import torch
    import uvicorn
    from qwen_tts import Qwen3TTSModel

    from uqt.engine import Engine

    t0 = time.time()
    torch.set_grad_enabled(False)
    tts = Qwen3TTSModel.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-1.7B-{args.variant}", device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    with torch.inference_mode():
        if args.mode == "single":
            eng = Engine(tts, max_len=2048)
            p = eng.build_prompt("custom", text="Warming up the engine for streaming.", language="English", speaker="Ryan")
            eng.warmup(p, Ks=(1, 2, 4, 8), max_bucket=1024, prefill_max=512)
            for _ in range(2):
                for _w in eng.stream(p, schedule=(1, 2, 4, 8)):
                    pass
            worker = SingleWorker(eng)
        else:
            from uqt.batch_engine import BatchEngine
            eng = Engine(tts, max_len=2048, streaming_codec=False, autotune=False)
            be = BatchEngine(eng, max_slots=args.max_slots, max_len=args.max_len)
            be.warmup(Ks=(2,))
            worker = BatchWorker(eng, be)
    worker.t.start()
    STATE["ready"] = True
    print(f"[uqt.server] ready in {time.time() - t0:.0f}s mode={args.mode}", flush=True)
    uvicorn.run(build_app(worker), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
