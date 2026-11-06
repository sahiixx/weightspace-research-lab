#!/usr/bin/env python3
"""Proposal-only Codex tool: does not sign or apply a patch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from wslab.checkpoints import load_checkpoint, write_json
from wslab.search import propose
from wslab.data import samples


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--workspace", type=Path, default=Path.cwd())
    p.add_argument("--proposal-name", default="candidate.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--count", type=int, default=2048)
    args = p.parse_args(argv)
    if not args.proposal_name.endswith(".json") or Path(args.proposal_name).name != args.proposal_name:
        p.error("Proposal must be a simple .json filename")
    output_root = (args.workspace.resolve() / "runs" / "agent-proposals").resolve()
    if any(t.is_symlink() for t in (args.workspace, args.workspace / "runs", args.workspace / "runs" / "agent-proposals")):
        p.error("No symlinked output directory allowed")
    patch, evidence = propose(load_checkpoint(args.checkpoint), samples("validation", args.seed, args.count))
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / args.proposal_name
    write_json(output, patch)
    write_json(output.with_suffix(".search.json"), evidence)
    print(json.dumps({"status": "unsigned_proposal_only", "authorized": False,
                      "patch": str(output), "search": str(output.with_suffix(".search.json"))}, indent=2))


if __name__ == "__main__":
    main()
