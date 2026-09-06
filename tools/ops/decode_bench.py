"""Decode speed of her served model: one short prompt, 256 new tokens,
tokens per second from the server's usage counts and wall time."""
import json
import time
import urllib.request

URL = "http://127.0.0.1:30000/v1/chat/completions"
MODEL = "jenny-qwen3.8-27b"
body = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Write a plain 300-word description of how a river shapes a valley over ten thousand years."}],
    "max_tokens": 256,
    "temperature": 0.0,
    "stream": False,
}
for run in range(3):
    request = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as reply:
        result = json.loads(reply.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    usage = result.get("usage", {})
    out = usage.get("completion_tokens", 0)
    print(f"run {run+1}: {out} tokens in {dt:.2f}s = {out/dt:.1f} tok/s (prompt {usage.get('prompt_tokens')})")
