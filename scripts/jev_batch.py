#!/usr/bin/env python3
"""Batch-answer a jev-questions-style file ({section: {state, questions}}).

`laya answer` (the Rust binary) only understands a single state plus its
questions (`{"state": ..., "questions": {qid: {...}}}`) in, and
`{qid: {...answer}}` out — it has no idea what a "section" is. This script
is the multi-section batch driver that used to live in the binary itself: it
calls `laya answer` once per section and assembles the results into
`{section: {qid: {...answer}}}`, matching what `laya jev` used to write
directly.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def find_default_binary() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    for candidate in ("target/release/laya", "target/debug/laya"):
        path = repo_root / candidate
        if path.is_file():
            return str(path)
    return "laya"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="jev-questions-style file: {section: {state, questions}}")
    parser.add_argument("output", help="where to write {section: {qid: {...answer}}} JSON")
    parser.add_argument("--model-dir", default="/mnt/extra/ai/laya/typed-decisions")
    parser.add_argument("--only", help="restrict every section to one question id")
    parser.add_argument(
        "--binary", default=None, help="path to the laya binary (default: auto-detect a build, else 'laya' on PATH)"
    )
    args = parser.parse_args()

    binary = args.binary or find_default_binary()

    sections = json.loads(Path(args.input).read_text())
    if not isinstance(sections, dict):
        sys.exit("expected top-level object of {section: {state, questions}}")

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for section, body in sections.items():
            if "state" not in body:
                sys.exit(f"{section}: missing state")
            if "questions" not in body:
                sys.exit(f"{section}: missing questions")

            questions = body["questions"]
            if args.only is not None:
                questions = {qid: q for qid, q in questions.items() if qid == args.only}
            if not questions:
                continue

            section_input = tmp_path / f"{section}.in.json"
            section_output = tmp_path / f"{section}.out.json"
            section_input.write_text(json.dumps({"state": body["state"], "questions": questions}))

            print(f"{section}: ", end="", file=sys.stderr, flush=True)
            cmd = [binary, "answer", str(section_input), str(section_output), "--model-dir", args.model_dir]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

            out[section] = json.loads(section_output.read_text())

    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
