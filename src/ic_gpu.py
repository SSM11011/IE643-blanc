"""Batched GPU inference for IndicConformer-600M CTC (encoder + CTC head converted from ONNX with onnx2torch)."""
import json, torch
from onnx2torch import convert
from common import MODELS
A = f"{MODELS}/indic-conformer-600m-multilingual/assets"
class ICGpu:
    def __init__(self):
        self.pre = torch.jit.load(f"{A}/preprocessor.ts", map_location="cuda")
        self.enc = convert(f"{A}/encoder.onnx").cuda().eval()
        self.ctc = convert(f"{A}/ctc_decoder.onnx").cuda().eval()
        self.vocab = json.load(open(f"{A}/vocab.json")); self.masks = json.load(open(f"{A}/language_masks.json")); self.blank = 256
    @torch.no_grad()
    def __call__(self, wavs, lang):
        """wavs: list of 1-D float32 numpy arrays (16 kHz) -> list of strings"""
        lens = torch.tensor([len(w) for w in wavs], device="cuda"); T = int(lens.max())
        x = torch.zeros(len(wavs), T, device="cuda")
        for i, w in enumerate(wavs): x[i, :len(w)] = torch.from_numpy(w).cuda()
        sig, slen = self.pre(input_signal=x, length=lens)
        out, elen = self.enc(sig, slen)                       # (B, 1024, T')
        lp = self.ctc(out)                                    # (B, T', V)
        lp = lp[:, :, self.masks[lang]].log_softmax(-1)
        idx = lp.argmax(-1).cpu(); elen = elen.cpu(); res = []
        for b in range(len(wavs)):
            seq = torch.unique_consecutive(idx[b, :int(elen[b])])
            res.append("".join(self.vocab[lang][t] for t in seq.tolist() if t != self.blank).replace("▁", " ").strip())
        return res
