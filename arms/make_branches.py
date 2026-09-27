#!/usr/bin/env python3
"""Derive the demo branches from main.

    arms/make_branches.py            build both, do not push
    arms/make_branches.py --push     build and force-push

Engine reads a repository at a ref. With all three variants in one tree behind
AGENT_VARIANT, it sees the fix sitting next to the bug and has nothing to
propose -- so each variant gets a branch where the other versions do not exist.

GENERATED, NEVER HAND-EDITED. Every edit is an anchored substitution that
raises if its anchor moved, so a refactor on main breaks the build loudly
instead of producing a branch that is quietly no longer the thing it claims.

  demo/naive      run_query returns every row. No receipt, no result tools, no
                  typed delegation. What a first prototype looks like.
  demo/regressed  the offload architecture with one real regression: the
                  receipt drops `distinct`, so "how many distinct aircraft"
                  has nothing to read.

main stays AHEAD of both: it is the fixed build, and a branch is main minus
something. Engine cannot pass by re-adding what a branch never had.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sh(*args: str, check: bool = True) -> str:
    out = subprocess.run(["git", "-C", str(ROOT), *args],
                         capture_output=True, text=True)
    if check and out.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()[:300]}")
    return out.stdout.strip()


def edit(path: str, old: str, new: str) -> None:
    """Anchored substitution. Raises if the anchor is gone or ambiguous."""
    p = ROOT / path
    s = p.read_text()
    if s.count(old) != 1:
        raise SystemExit(f"{path}: anchor found {s.count(old)} times, expected 1:\n  {old[:90]}")
    p.write_text(s.replace(old, new))


def drop(path: str) -> None:
    (ROOT / path).unlink(missing_ok=True)


# --- demo/naive -------------------------------------------------------------

def build_naive() -> None:
    """Remove the offload architecture entirely."""
    drop("src/agent/subagents/analyst.py")
    drop("src/agent/middleware/readable_answer.py")

    # One tool, and it returns rows.
    p = ROOT / "src/agent/tools/query.py"
    s = p.read_text()
    keep_to = s.index("# --- v2: receipt plus targeted tools")
    p.write_text(s[:keep_to] + 'TOOLS = [run_query_inline]\nV1_TOOLS = TOOLS\n')

    # results.py keeps only the database call. No store, no index, no breakdown.
    p = ROOT / "src/agent/results.py"
    s = p.read_text()
    head = s[:s.index("@dataclass")]
    query_fn = s[s.index("def query(sql: str"):s.index("def save(")]
    p.write_text(head.replace(
        '"""Where query rows live instead of in the conversation.',
        '"""Running the approved SQL.') + query_fn)

    edit("src/agent/agent.py",
         "from middleware.readable_answer import readable_answer\nfrom middleware.run_cost import RunCost, run_cost\n"
         "from tools.query import V1_TOOLS, V2_TOOLS",
         "from middleware.run_cost import RunCost, run_cost\nfrom tools.query import V1_TOOLS")
    edit("src/agent/agent.py",
         "from subagents.analyst import analyze_result\n", "")
    edit("src/agent/agent.py",
         '    tools = list(V1_TOOLS) if v == "v1" else [*V2_TOOLS, analyze_result]',
         "    tools = list(V1_TOOLS)")
    edit("src/agent/agent.py",
         "        middleware=[readable_answer, run_cost],", "        middleware=[run_cost],")
    # The V2 prompt described tools that no longer exist.
    p = ROOT / "src/agent/agent.py"
    s = p.read_text()
    p.write_text(s[:s.index("V2_PROMPT = _SHARED")] +
                 'V2_PROMPT = V1_PROMPT\n\n' + s[s.index("def build_agent("):])
    drop("docs/architecture.md")


# --- demo/regressed ---------------------------------------------------------

def build_regressed() -> None:
    """Keep the architecture; drop `distinct` from the receipt.

    The actual regression from commit 15b7868, preserved: collapsing the column
    summaries dropped high-cardinality columns entirely, and with them the
    cardinality. Nothing in review caught it.
    """
    edit("src/agent/results.py",
         "        breakdown=breakdown, distinct={} if OMIT_DISTINCT else distinct,",
         "        breakdown=breakdown, distinct={},")
    edit("src/agent/agent.py",
         '  distinct    "how many distinct aircraft are affected" \u2014 the cardinality of a\n'
         "              column too varied to break down, already counted for you\n", "")


BRANCHES = {"demo/naive": build_naive, "demo/regressed": build_regressed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    if sh("status", "--porcelain"):
        raise SystemExit("working tree is dirty; commit or stash first")
    start = sh("rev-parse", "--abbrev-ref", "HEAD")
    if start != "main":
        raise SystemExit(f"run this from main (on {start})")

    for branch, build in BRANCHES.items():
        sh("checkout", "--quiet", "-B", branch, "main")
        try:
            build()
        except SystemExit:
            # A failed build leaves the tree half-edited. Reset it and get off
            # the branch before re-raising, or those edits ride back to main on
            # the next checkout -- which happened, and put the regression this
            # script is supposed to CREATE onto main, where it was committed.
            sh("checkout", "--quiet", "--", ".")
            sh("clean", "--quiet", "-fd")
            sh("checkout", "--quiet", "main")
            raise
        sh("add", "-A")
        sh("-c", "user.email=demo@fleet-sql.local", "-c", "user.name=fleet-sql",
           "commit", "--quiet", "-m",
           f"{branch}: derived from main by arms/make_branches.py\n\n"
           f"Generated, not hand-edited. Rebuild with arms/make_branches.py.")
        changed = sh("diff", "--name-only", "main", branch).splitlines()
        print(f"  {branch}: {len(changed)} file(s) differ from main")
        for f in changed:
            print(f"      {f}")
        # A branch that differs from main in nothing has not been built.
        if not changed:
            raise SystemExit(f"{branch} is identical to main")
        sh("checkout", "--quiet", "main")

    if args.push:
        for branch in BRANCHES:
            sh("push", "--quiet", "--force-with-lease", "origin", branch)
        print(f"  pushed: {', '.join(BRANCHES)}")
    else:
        print("\n  not pushed (use --push)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
