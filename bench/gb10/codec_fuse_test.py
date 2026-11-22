"""Fused residual-unit codec path vs the unfused one on the box: waveform SNR over consecutive
streaming calls (state carried, mixed new/continuing rows) and graphed time per stream-frame.
Optionally sweeps tile configs per width.

    python -m bench.gb10.codec_fuse_test --cases 64x2,256x2 --sweep
"""
import argparse
import itertools
import json

import torch


def snr(ref, got):
    ref, got = ref.double(), got.double()
    return float(10 * torch.log10(ref.pow(2).sum() / (ref - got).pow(2).sum().clamp(min=1e-30)))


def time_graph(sd, B, F, iters=10):
    from uqt.streaming_codec import StreamingDecodeGraphs
    codes = torch.randint(0, 2048, (B, F, 16), device="cuda", dtype=torch.int32)
    slots = torch.arange(B, device="cuda", dtype=torch.int32)
    pos = torch.full((B,), 40, device="cuda", dtype=torch.int32)
    g = StreamingDecodeGraphs(sd, [B], frames=F)
    for _ in range(3):
        g(codes, slots, pos)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        g(codes, slots, pos)
    en.record()
    torch.cuda.synchronize()
    del g
    return st.elapsed_time(en) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="64x2,256x2")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep-trans", action="store_true")
    ap.add_argument("--out", default="/root/results/codec_fuse_test.json")
    args = ap.parse_args()
    from qwen_tts import Qwen3TTSModel

    import uqt.streaming_codec as sc

    torch.set_grad_enabled(False)
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    dec = tts.model.speech_tokenizer.model.decoder
    res = {}
    with torch.inference_mode():
        ref = sc.StreamingCodecDecoder(dec, num_slots=256)
        ref.res_unit = False
        ref.fused_trans = False
        new = sc.StreamingCodecDecoder(dec, num_slots=256)
        new.res_unit = True
        f32 = sc.StreamingCodecDecoder(dec.float(), num_slots=64, dtype=torch.float32)
        dec.to(torch.bfloat16)
        # exactness: 4 consecutive calls of F frames; rows 0..B-1, half start fresh at call 2
        torch.manual_seed(0)
        B = 48
        exact = []
        for F in (1, 2, 5):
            pos = torch.zeros(B, device="cuda", dtype=torch.int32)
            slots = torch.arange(B, device="cuda", dtype=torch.int32)
            for call in range(4):
                if call == 2:
                    pos[: B // 2] = 0
                codes = torch.randint(0, 2048, (B, F, 16), device="cuda", dtype=torch.int32)
                wr = ref(codes, slots, pos)
                wn = new(codes, slots, pos)
                wf = f32(codes, slots, pos)
                exact.append({"F": F, "call": call, "snr_db": round(snr(wr, wn), 2),
                              "ref_vs_f32": round(snr(wf, wr), 2), "fused_vs_f32": round(snr(wf, wn), 2),
                              "maxabs": float((wr - wn).abs().max())})
                pos += F
        res["exact"] = exact
        print("exact", json.dumps(exact), flush=True)
        for case in args.cases.split(","):
            B, F = (int(x) for x in case.split("x"))
            tr = time_graph(ref, B, F)
            tn = time_graph(new, B, F)
            res[case] = {"ref_ms": tr, "fused_ms": tn, "ref_us_sf": tr * 1e3 / (B * F),
                         "fused_us_sf": tn * 1e3 / (B * F)}
            print(case, json.dumps(res[case]), flush=True)
        if args.sweep:
            B, F = 128, 4
            sw = {}
            base = dict(sc._RES_UNIT)
            for c in (96, 192, 384):
                best = None
                for bm, bc, w, stg in itertools.product((32, 64, 128), (32, 64), (4, 8), (2, 3)):
                    if c % bc or (c == 384 and bm == 128 and w == 4):
                        continue
                    sc._RES_UNIT.clear()
                    sc._RES_UNIT.update(base)
                    sc._RES_UNIT[c] = (bm, bc, w, stg)
                    try:
                        ms = time_graph(new, B, F, iters=5)
                    except Exception as e:  # noqa: BLE001
                        print(c, (bm, bc, w, stg), "fail", repr(e)[:120], flush=True)
                        continue
                    print(c, (bm, bc, w, stg), round(ms, 3), flush=True)
                    if best is None or ms < best[1]:
                        best = ((bm, bc, w, stg), ms)
                base[c] = best[0]
                sw[c] = best
            sc._RES_UNIT.clear()
            sc._RES_UNIT.update(base)
            res["sweep_best"] = {str(k): v for k, v in sw.items()}
            print("best", sw, flush=True)
        if args.sweep_trans:
            B, F = 128, 4
            sw = {}
            base = dict(sc._TRANS)
            for c in (96, 192, 384, 768):
                best = None
                for bm, bn, bk, w, stg in itertools.product((64, 128), (32, 64, 128), (32, 64), (4, 8), (2, 3)):
                    if bn > c or (bm * bk + bk * bn) * 2 * stg > 96000:
                        continue
                    sc._TRANS.clear()
                    sc._TRANS.update(base)
                    sc._TRANS[c] = (bm, bn, bk, w, stg)
                    try:
                        ms = time_graph(new, B, F, iters=5)
                    except Exception as e:  # noqa: BLE001
                        print("T", c, (bm, bn, bk, w, stg), "fail", repr(e)[:100], flush=True)
                        continue
                    print("T", c, (bm, bn, bk, w, stg), round(ms, 3), flush=True)
                    if best is None or ms < best[1]:
                        best = ((bm, bn, bk, w, stg), ms)
                base[c] = best[0]
                sw[c] = best
            sc._TRANS.clear()
            sc._TRANS.update(base)
            res["sweep_trans_best"] = {str(k): v for k, v in sw.items()}
            print("best_trans", sw, flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
