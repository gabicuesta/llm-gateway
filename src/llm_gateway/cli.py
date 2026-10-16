"""`llm-gateway serve | key | check`"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import load_config
from .teams import new_key


def _load_env_file(path: str = ".env") -> None:
    if not os.path.isfile(path):
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def app_from_env():
    """Factory for `uvicorn --factory llm_gateway.cli:app_from_env`."""
    from .app import create_app

    _load_env_file()
    return create_app(load_config(os.environ.get("LLM_GATEWAY_CONFIG", "config.yaml")))


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-gateway")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run the gateway")
    serve.add_argument("--config", default="config.yaml")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    key = sub.add_parser("key", help="Create a virtual API key")
    key.add_argument("--team", required=True)

    check = sub.add_parser("check", help="Validate a config file")
    check.add_argument("--config", default="config.yaml")

    args = parser.parse_args()
    _load_env_file()

    if args.cmd == "key":
        plain, hashed = new_key()
        print(f"Key for '{args.team}' (give it to the team, it is not stored):\n  {plain}\n")
        print(f"Add this hash to the team's key_hashes in config.yaml:\n  {hashed}")
        return

    try:
        config = load_config(args.config)
    except Exception as exc:  # show config errors plainly
        print(f"Invalid config: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.cmd == "check":
        missing = [f"{n} ({p.api_key_env})" for n, p in config.providers.items() if p.api_key_env and not p.api_key]
        print(f"OK · {len(config.models)} aliases · {len(config.teams)} teams · {len(config.providers)} providers")
        if missing:
            print(f"Warning: no API key in the environment for: {', '.join(missing)}")
        return

    import uvicorn

    from .app import create_app

    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
