"""A fake llama-server for tests (only used by tests).

Implements ``/health``, ``/tokenize`` (``<|im_start|>`` is one special token
when ``parse_special``), ``/completion`` (letter ``top_logprobs``), and
``/props``. The answer favors the option whose description contains one of
``favor`` keywords (first option otherwise). Behaviors are configurable:
``delay_s`` per completion, ``missing_below`` (omit the favored letter unless
``n_probs`` ≥ that value), ``n_ctx`` and ``health_status``.

Run as a script (``python fake_llama.py --port N ...``) for service-lifecycle
tests that need a real child process.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LETTERS = "ABCDEFGHIJKLMNOP"


class FakeConfig:
    def __init__(self, **kw):
        self.favor = kw.get("favor", ["misdelivered", "false"])
        self.delay_s = kw.get("delay_s", 0.0)
        self.missing_below = kw.get("missing_below", 0)
        self.n_ctx = kw.get("n_ctx", 8192)
        self.health_status = kw.get("health_status", 200)
        self.tokens_per_char = kw.get("tokens_per_char", 0.25)
        self.completions = 0
        self.n_probs_seen: list[int] = []
        self.lock = threading.Lock()
        self.prompts: dict[int, str] = {}


def _options(prompt: str) -> list[str]:
    match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.S)
    if not match:
        return []
    try:
        payload = json.loads(match.group(1))
    except ValueError:
        return []
    return [o["description"] for o in payload.get("options", [])]


def make_handler(cfg: FakeConfig):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                if cfg.health_status == 200:
                    self._send(200, {"status": "ok"})
                else:
                    self._send(cfg.health_status, {"error": {"message": "Loading model"}})
            elif self.path == "/props":
                self._send(200, {"default_generation_settings": {"n_ctx": cfg.n_ctx}})
            else:
                self._send(404, {})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/tokenize":
                content = body.get("content", "")
                if content == "<|im_start|>" and body.get("parse_special"):
                    self._send(200, {"tokens": [151644]})
                    return
                with cfg.lock:
                    key = len(cfg.prompts) + 1
                    cfg.prompts[key] = content
                count = max(1, int(len(content) * cfg.tokens_per_char))
                self._send(200, {"tokens": [key] + [0] * (count - 1)})
            elif self.path == "/completion":
                if cfg.delay_s:
                    time.sleep(cfg.delay_s)
                with cfg.lock:
                    cfg.completions += 1
                    n_probs = int(body.get("n_probs", 0))
                    cfg.n_probs_seen.append(n_probs)
                    prompt = cfg.prompts.get(body.get("prompt", [0])[0], "")
                options = _options(prompt)
                favored = 0
                for i, text in enumerate(options):
                    if any(word in text for word in cfg.favor):
                        favored = i
                        break
                top = []
                for i, _ in enumerate(options or ["x"]):
                    if i == favored and n_probs < cfg.missing_below:
                        continue
                    top.append({"token": LETTERS[i], "logprob": -0.05 if i == favored else -4.0})
                top.append({"token": " the", "logprob": -9.0})
                self._send(200, {
                    "completion_probabilities": [{"top_logprobs": top}],
                    "tokens_evaluated": len(prompt) // 4,
                    "timings": {"prompt_ms": 1.0, "predicted_ms": 1.0},
                })
            else:
                self._send(404, {})

        def log_message(self, *args):
            pass

    return Handler


class FakeLlamaServer:
    def __init__(self, **kw):
        self.cfg = FakeConfig(**kw)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.cfg))
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("-m", dest="model")
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--parallel")
    parser.add_argument("--n-gpu-layers")
    args, _ = parser.parse_known_args()
    cfg = FakeConfig(n_ctx=args.ctx_size)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(cfg))
    server.serve_forever()


if __name__ == "__main__":
    main()
