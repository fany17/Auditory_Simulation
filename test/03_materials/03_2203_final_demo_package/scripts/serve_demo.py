#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--bind", default="127.0.0.1")
parser.add_argument("--root", type=Path, default=ROOT)
args = parser.parse_args()
demo = (args.root / "demo" / "TB001-DEMO001").resolve()
if not (demo / "index.html").is_file():
    raise SystemExit(f"Missing demo entry: {demo / 'index.html'}")
handler = functools.partial(SimpleHTTPRequestHandler, directory=str(demo))
server = ThreadingHTTPServer((args.bind, args.port), handler)
print(f"Serving {demo} at http://{args.bind}:{args.port}/")
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    server.server_close()
