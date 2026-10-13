"""Load a vanilla, LoRA-fine-tuned or fully fine-tuned Whisper for evaluation (merged, bf16, eval mode, on GPU).

Odia: if the adapter was trained with the new <|or|> token, the base embedding is resized to vocab+1 BEFORE loading the adapter
(PEFT saves the resized embed_tokens / proj_out). A fully fine-tuned model saved with the token has vocab_size = len(tokenizer) + 1.
In both cases LANGID['or'] = len(tokenizer).
NOTE: tok.convert_tokens_to_ids('<|or|>') silently returns EOT (50257) - always use the LANGID returned here."""
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_LANGS = ["hi", "en", "te", "as", "bn", "mr"]


def load(model_path, adapter="", dtype=torch.bfloat16):
    proc = WhisperProcessor.from_pretrained(model_path)
    tok = proc.tokenizer
    m = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)
    LANGID = {l: tok.convert_tokens_to_ids(f"<|{l}|>") for l in BASE_LANGS}
    if adapter:
        from safetensors import safe_open
        from peft import PeftModel
        with safe_open(f"{adapter}/adapter_model.safetensors", "pt") as f:
            emb_keys = [k for k in f.keys() if k.endswith("embed_tokens.weight")]
            n_vocab = f.get_slice(emb_keys[0]).get_shape()[0] if emb_keys else m.config.vocab_size
        if n_vocab == m.config.vocab_size + 1:
            m.resize_token_embeddings(n_vocab, mean_resizing=False)
            m.config.vocab_size = n_vocab
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    if m.config.vocab_size == len(tok) + 1:
        LANGID["or"] = len(tok)
    m = m.cuda().eval()
    m.generation_config.forced_decoder_ids = None
    m.generation_config.max_length = None
    return proc, tok, m, LANGID
