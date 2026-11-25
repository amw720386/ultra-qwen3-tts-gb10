# Running on a GB10

Everything runs directly on the box. Setup quirks:

```bash
# container: NGC PyTorch 25.10 (nvcr.io/nvidia/pytorch:25.10-py3)
pip uninstall -y torchao
pip install --no-deps torchaudio==2.9.0     # --no-deps keeps the NGC torch build
export UQT_NO_PDL=1                         # required with the container's Triton 3.4
```

Benchmarks (from the repository root):

```bash
# single-stream timings + batched step-time scan
python -m bench.gb10.bench_local --variant CustomVoice --weights fp8 --batch 1,8,32,64,128,256

# accuracy: 300 texts per weight format, Whisper large-v3 WER
python -m bench.gb10.acc_local --weights bf16,fp8 --n 300 --manifest /root/manifest.json

# codec decoder: fused vs unfused path (waveform SNR + time per stream-frame)
python -m bench.gb10.codec_fuse_test --cases 64x2,256x2

# streaming HTTP head-to-head on the box (engine process + Go gateway, or a running competitor server)
python -m bench.gb10.h2h_local --system uqt --weights fp8
```

The accuracy manifest is the LibriSpeech test-clean set built by `bench/baseline.py::prep`. Further
profilers and microbenchmarks are in `bench/gb10/`; measured outputs are in `results/gb10/`.

Serve it: build the gateway once, then start the engine process and the gateway.

```bash
(cd gateway && go build -o uqt-gw .)
python -m uqt.engine_proc --sock /tmp/uqt.sock --max-slots 256 --max-len 512 --weights hyb
./gateway/uqt-gw -sock /tmp/uqt.sock -addr 127.0.0.1:8091
curl -s http://127.0.0.1:8091/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"input": "Hello from one GB10.", "voice": "Ryan", "language": "English"}' > out.pcm   # s16le, 24 kHz, mono
```

The first start compiles and captures every kernel and graph, and with hybrid weights it takes many minutes. The server
answers `GET /health` once it is warm.
