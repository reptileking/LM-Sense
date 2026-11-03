"""``python -m senselm.codec --config configs/codec.yaml encode|decode|roundtrip TEXT_OR_IDS``"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from senselm.codec.api import Codec
from senselm.common import load_config, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("command", choices=["encode", "decode", "roundtrip"])
    ap.add_argument("text", help="text to encode, or a JSON list of ids to decode")
    args = ap.parse_args()
    setup_logging("WARNING")
    codec = Codec(load_config(args.config), encoder=args.command != "decode")
    if args.command == "decode":
        print(codec.decode(json.loads(args.text)))
        return
    ids = codec.encode(args.text)
    print(" ".join(codec.names(ids)))
    print(f"{len(ids)} tokens")
    if args.command == "roundtrip":
        back = codec.decode(ids)
        print(repr(back))
        print("exact" if back == args.text else "MISMATCH")


if __name__ == "__main__":
    main()
