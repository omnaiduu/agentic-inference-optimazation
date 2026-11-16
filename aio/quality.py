"""Accuracy tests that work for ANY engine behind /v1/completions (ours and vLLM).

1. Token agreement (teacher-forced): walk through the oracle's own answers token by token and
   ask the engine "what would you predict next here?". Unlike comparing whole texts, one early
   near-tie cannot cascade: every position is judged on its own. 1.0 = identical model.
2. GSM8K-100: 100 grade-school math word problems, greedy, exact final answer.
   This answers "did compression make the model dumber?", not "are the words identical?".
"""
import asyncio
import json
import re
import urllib.request

import aiohttp


def _post(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def token_agreement(url: str, ref: dict, model: str, engine: str) -> dict:
    hits = total = 0
    for case in ref["cases"]:
        p, t = case["prompt_ids"], case["output_ids"]
        if engine == "ours":
            preds = _post(f"{url}/aio/teacher_forced", {"prompt_ids": p, "target_ids": t})["preds"]
            ok = [a == b for a, b in zip(preds, t)]
        else:  # vLLM: score the whole sequence once and read the rank of every oracle token
            r = _post(f"{url}/v1/completions", {"model": model, "prompt": p + t, "max_tokens": 1,
                                                 "temperature": 0.0, "prompt_logprobs": 1})
            plp = r["choices"][0]["prompt_logprobs"]
            ok = [plp[len(p) + j][str(tok)]["rank"] == 1 for j, tok in enumerate(t)]
        hits += sum(ok)
        total += len(ok)
    print(f"  token agreement with oracle: {hits}/{total} = {hits / total:.1%}", flush=True)
    return {"hits": hits, "total": total, "agreement": hits / total}


def _number(s: str):
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", s) or re.findall(r"(-?[\d,]*\.?\d+)", s)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def gsm8k(url: str, model: str, tokenizer_dir: str, n: int = 100, max_tokens: int = 400) -> dict:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    path = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path).head(n)
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": q + "\nSolve it step by step. On the last line write the final "
                                         "answer as '#### <number>'."}],
        add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"] for q in df.question]
    golds = [_number(a.split("####")[-1]) for a in df.answer]

    async def run():
        sem = asyncio.Semaphore(16)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
            async def one(p):
                async with sem:
                    body = {"model": model, "prompt": [int(i) for i in p], "max_tokens": max_tokens,
                            "temperature": 0.0}
                    async with s.post(f"{url}/v1/completions", json=body) as r:
                        return (await r.json())["choices"][0]["text"]
            return await asyncio.gather(*[one(p) for p in prompts])

    outs = asyncio.run(run())
    correct = [(_number(o) is not None and g is not None and abs(_number(o) - g) < 1e-6)
               for o, g in zip(outs, golds)]
    acc = sum(correct) / len(correct)
    print(f"  GSM8K-{n}: {sum(correct)}/{len(correct)} = {acc:.0%}", flush=True)
    return {"n": n, "correct": sum(correct), "accuracy": acc, "correct_list": correct,
            "samples": [o[-200:] for o in outs[:3]]}
