"""Shared helpers for the training / eval scripts."""
import json, torch


def neutralise_torchao():
    """peft>=0.18 raises if an *incompatible* torchao is merely installed (Kaggle image ships 0.10.0).
    We never use torchao; make peft believe it is absent, in every module that bound the check."""
    try:
        import peft.import_utils as iu
        iu.is_torchao_available = lambda: False
        import importlib
        for mod in ("peft.tuners.lora.torchao", "peft.tuners.lora.model", "peft.tuners.tuners_utils"):
            try:
                m = importlib.import_module(mod)
                if hasattr(m, "is_torchao_available"): m.is_torchao_available = lambda: False
            except Exception:
                pass
    except Exception:
        pass


neutralise_torchao()


def read_items(files, limit=None, tasks=None):
    """limit applies PER FILE (a global head() silently dropped every SGA item in pilot-4)."""
    items = []
    for f in files:
        n = 0
        with open(f) as fh:
            for line in fh:
                it = json.loads(line)
                if tasks and it["task"] not in tasks: continue
                items.append(it); n += 1
                if limit and n >= limit: break
        print(f"[data] {f}: {n} items", flush=True)
    return items


def dtype_and_device():
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        return (torch.bfloat16 if major >= 8 else torch.float16), "cuda"
    return torch.float32, "cpu"


def load_model_and_tok(model_path, adapter=None, trainable_adapter=False, attn=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt, dev = dtype_and_device()
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = dict(dtype=dt, device_map=dev)
    if attn: kw["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable_adapter)
    return model, tok, dt


def chat_kwargs(tok):
    """Qwen3 templates accept enable_thinking; other templates ignore unknown kwargs badly, so probe."""
    try:
        tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return {"enable_thinking": False}
    except Exception:
        return {}
