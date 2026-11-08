#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub==1.33.0"]
# ///

from __future__ import annotations

import hashlib
import runpy
import sys
from pathlib import Path

from huggingface_hub import cached_assets_path, hf_hub_download, snapshot_download

REPOSITORY = "alvarobartt/sys1-benchmarks"
REVISION = "a3c3616418ef2d20d2b39bfe8d0c2c905305df5d"
CLIENT_SHA256 = "fc61fe89d3b37e6d25cee268565a496bdda80b831fc44ed5676b4c82bcd00189"
ROWS_SHA256 = "3cc90c88b6f81317f275bee7c2e2292ffc9acd3f581d76a5108acce29b0b743e"
WARMUP_SHA256 = "428daff6f9522a6c9b73c5329bc5ade571306c91b7adb71213b5e0ad6ff2e800"


def main() -> None:
    arguments = sys.argv[1:]
    supplied = {argument.split("=", 1)[0] for argument in arguments}
    if "--help" in arguments or "-h" in arguments:
        client = Path(hf_hub_download(REPOSITORY, "frozen/leaderboard.py",
                                    repo_type="dataset", revision=REVISION))
    else:
        root = Path(snapshot_download(REPOSITORY, repo_type="dataset", revision=REVISION,
                                      local_dir=cached_assets_path("sys1", "leaderboard", REVISION),
                                      allow_patterns=["frozen/leaderboard.py", "frozen/inputs/release-cuda/**"]))
        client = root / "frozen/leaderboard.py"
        inputs = root / "frozen/inputs/release-cuda"
        for flag, path, pin_flag, pin in [
            ("--rows", inputs / "cases/canonical.jsonl", "--expected-rows-sha256", ROWS_SHA256),
            ("--warmup", inputs / "warmup.jsonl", "--expected-warmup-sha256", WARMUP_SHA256),
        ]:
            if flag not in supplied:
                arguments.extend([flag, str(path)])
                if pin_flag not in supplied:
                    arguments.extend([pin_flag, pin])
        if "--protocol" not in supplied:
            arguments.extend(["--protocol", "matched-supported"])
        stress = any(flag.startswith(("--stress-", "--arrival-")) for flag in supplied)
        if "--expected-stress-manifest-sha256" in supplied:
            stress = True
        if stress and "--stress-manifest" not in supplied:
            arguments.extend(["--stress-manifest", str(inputs / "manifest.json")])
            if "--expected-stress-manifest-sha256" not in supplied:
                arguments.extend(["--expected-stress-manifest-sha256",
                                  "dde3123d1966b5ca105ebe85e0f01ade24b631ac0e016698a751a7e736ab697a"])
    if hashlib.sha256(client.read_bytes()).hexdigest() != CLIENT_SHA256:
        raise RuntimeError("benchmark client SHA256 mismatch")
    sys.argv = [str(client), *arguments]
    runpy.run_path(str(client), run_name="__main__")


if __name__ == "__main__":
    main()
