"""Shared test harness: wait for a server, and check its answers against the oracle."""
import json
import subprocess
import time
import urllib.request

CHECK_MAX_TOKENS = 64


def wait_ready(url: str, proc: subprocess.Popen, timeout_s: int = 1200) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{url}/health", timeout=2)
            return time.time() - t0
        except Exception:
            time.sleep(2)
    raise TimeoutError("server did not become ready")


def check_correctness(url: str, ref: dict, model: str) -> dict:
    """Greedy-decode the oracle prompts on the engine and compare text with the oracle."""
    rows = []
    for case in ref["cases"]:
        body = json.dumps({"model": model, "prompt": case["prompt_ids"], "max_tokens": CHECK_MAX_TOKENS,
                           "temperature": 0.0}).encode()
        req = urllib.request.Request(f"{url}/v1/completions", body, {"Content-Type": "application/json"})
        got = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["text"]
        want = case["output_text"]
        prefix = 0
        while prefix < min(len(got), len(want)) and got[prefix] == want[prefix]:
            prefix += 1
        rows.append({"exact": got == want, "common_prefix_frac": prefix / max(len(want), 1),
                     "got": got[:160], "want": want[:160]})
    n_exact = sum(r["exact"] for r in rows)
    avg_prefix = sum(r["common_prefix_frac"] for r in rows) / len(rows)
    print(f"  correctness: {n_exact}/{len(rows)} exact matches, avg common prefix {avg_prefix:.0%}", flush=True)
    for r in rows:
        if not r["exact"]:
            print(f"    DIFF after {r['common_prefix_frac']:.0%}:\n      got : {r['got']!r}\n      want: {r['want']!r}")
    return {"exact": n_exact, "total": len(rows), "avg_common_prefix": avg_prefix, "rows": rows}
