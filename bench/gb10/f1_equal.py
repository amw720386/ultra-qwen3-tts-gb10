"""One-frame first chunks must decode to the same audio as a two-frame chunk: codes decoded as 2 vs 1 + 1.

    python -m bench.gb10.f1_equal
"""
import torch


def main():
    from qwen_tts import Qwen3TTSModel

    from uqt.streaming_codec import StreamingCodecDecoder, StreamingDecodeGraphs

    torch.set_grad_enabled(False)
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    with torch.inference_mode():
        sdec = StreamingCodecDecoder(tts.model.speech_tokenizer.model.decoder, num_slots=32)
        torch.manual_seed(1)
        n = 16
        codes = torch.randint(0, 2000, (n, 8, 16), device="cuda")
        sl = torch.arange(n, dtype=torch.int32, device="cuda")
        g2 = StreamingDecodeGraphs(sdec, [n], frames=2)
        g1 = StreamingDecodeGraphs(sdec, [n], frames=1, pool=g2.pool)

        def run(plan):
            out, p = [], 0
            for f in plan:
                pos = torch.full((n,), p, dtype=torch.int32, device="cuda")
                out.append((g1 if f == 1 else g2)(codes[:, p:p + f].contiguous(), sl, pos).clone())
                p += f
            return torch.cat(out, 1).float()

        a, a2, b = run([2, 2, 2, 2]), run([2, 2, 2, 2]), run([1, 1, 2, 1, 1, 2])
        for name, x in (("repeat", a2), ("1+1", b)):
            d = (x - a).abs()
            print(name, "max abs diff", float(d.max()), "SNR dB",
                  float(10 * torch.log10(a.pow(2).mean() / d.pow(2).mean().clamp_min(1e-20))), flush=True)


if __name__ == "__main__":
    main()
