"""Runtime probe for compose healthchecks (HTTP-only, no heavy deps)."""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request


def probe_http_ready(*, host: str, port: int, path: str = "/ready", timeout: float = 2.0) -> bool:
    url = f"http://{host}:{port}{path}"
    request = urllib.request.Request(url, headers={"Connection": "close"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            response.read()
            return status == 200
    except (urllib.error.URLError, OSError):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runtime probe for healthchecks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    http_parser = subparsers.add_parser("http-ready", help="Probe an HTTP readiness endpoint")
    http_parser.add_argument("--host", default="127.0.0.1")
    http_parser.add_argument("--port", type=int, required=True)
    http_parser.add_argument("--path", default="/ready")
    http_parser.add_argument("--timeout", type=float, default=2.0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    ok = probe_http_ready(host=args.host, port=args.port, path=args.path, timeout=args.timeout)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
