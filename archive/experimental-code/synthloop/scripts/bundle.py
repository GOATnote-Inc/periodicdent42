from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests

API = os.environ.get("SYNTH_API", "http://localhost:8080")
AUTH = (os.environ.get("BASIC_AUTH_USER", "admin"), os.environ.get("BASIC_AUTH_PASS", "changeme"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--dest")
    args = parser.parse_args()
    res = requests.get(f"{API}/bundles/{args.run}", auth=AUTH)
    res.raise_for_status()
    dest = Path(args.dest or f"{args.run}_bundle.zip")
    dest.write_bytes(res.content)
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()

