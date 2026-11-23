"""Rounding error of each KV cache format on the real talker cache.

For each text: prefill, generate frames, then rerun one talker step from the same state with the cache
round-tripped through a format, and compare the next-token distribution with the untouched cache.
"control" reruns with the untouched cache (must be 0).

    python -m bench.gb10.kv_real_error
"""
import json

import torch
from qwen_tts import Qwen3TTSModel

from uqt import kernels
from uqt.engine import Engine


def int8_rt(x):
    q, s = kernels.quant_rows(x)
    return (q.float() * s[..., None]).to(x.dtype)


def fp8_rt(x):
    return x.to(torch.float8_e4m3fn).to(x.dtype)


def same(x):
    return x


FORMATS = {"control": (same, same), "fp8": (fp8_rt, fp8_rt), "int8": (int8_rt, int8_rt),
           "fp8_k_only": (fp8_rt, same), "fp8_v_only": (same, fp8_rt),
           "int8_k_only": (int8_rt, same), "int8_v_only": (same, int8_rt)}


def main():
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    man = json.load(open("/root/manifest.json"))
    acc = {f: {"k_rel": [], "v_rel": [], "kl": [], "top1": []} for f in FORMATS}
    with torch.inference_mode():
        E = Engine(tts, max_len=2048, streaming_codec=False, autotune=False, weights="bf16")
        for n, it in enumerate(man[:24]):
            p = E.build_prompt("custom", text=it["target_text"], language="English", speaker="Ryan")
            E.prefill(p)
            E.step_frames(24, use_graph=False)
            E._frame_cp()  # next talker input in E.inp
            T = E.pos_host
            snap = {a: getattr(E, a).clone() for a in ("code0", "hidden", "pos", "step", "fidx", "seen")}
            k0, v0 = E.kc[:, :, :T].clone(), E.vc[:, :, :T].clone()
            ref = None
            for f, (fk, fv) in [("control", FORMATS["control"])] + list(FORMATS.items()):
                for a, t in snap.items():
                    getattr(E, a).copy_(t)
                kq, vq = fk(k0), fv(v0)
                E.kc[:, :, :T].copy_(kq)
                E.vc[:, :, :T].copy_(vq)
                E._frame_talker(E._bucket(T + 1))
                lg = torch.nn.functional.linear(E.hidden.float(), E.head.float())
                lp = torch.log_softmax(lg[0], -1)
                if ref is None:
                    ref = lp
                    continue
                d = acc[f]
                d["k_rel"].append(float((kq.float() - k0.float()).norm(dim=-1).div(k0.float().norm(dim=-1).clamp_min(1e-6)).mean()))
                d["v_rel"].append(float((vq.float() - v0.float()).norm(dim=-1).div(v0.float().norm(dim=-1).clamp_min(1e-6)).mean()))
                d["kl"].append(float((ref.exp() * (ref - lp)).sum()))
                d["top1"].append(float(ref.argmax() == lp.argmax()))
    out = {f: {k: sum(v) / len(v) for k, v in d.items()} | {"kl_max": max(d["kl"])} for f, d in acc.items()}
    for f, r in out.items():
        print(f, json.dumps({k: round(x, 5) for k, x in r.items()}), flush=True)
    json.dump(out, open("/root/results/kv_real_error.json", "w"), indent=1)


if __name__ == "__main__":
    main()
