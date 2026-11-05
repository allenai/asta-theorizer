"""Contract test: nora a2a_handler's theorizer transform → theorizer DataPart.

This is the UI-safety guard. The Nora UI sends a plain TextPart (the user's
research question); the a2a_handler's `transforms/theorizer.py` reshapes it into
the `literature-theory-generation` DataPart that the theorizer expects.

We assert the contract the theorizer's paper_store change must not break:
  1. The transform emits a single DataPart with kind=literature-theory-generation.
  2. theory_query == the UI text.
  3. **No `paper_store` key is ever emitted by the UI path.**  ← the key fact:
     the paper_store fix (isinstance guard in inject_papers) only fires when a
     caller supplies paper_store; the UI never does, so the UI path is immune.
  4. Defaults (max_papers_to_retrieve, do_qualified_novelty_evaluation) apply
     and are overridable via skill params.

Run (uses the theorizer venv, which has the `a2a` package):
    .venv/bin/python tests/harness/transform_contract_test.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

NORA_TRANSFORM = os.path.expanduser(
    "~/workspace/nora-v3/nora/a2a_handler/app/transforms/theorizer.py"
)


def _load_transform():
    spec = importlib.util.spec_from_file_location("nora_theorizer_transform", NORA_TRANSFORM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ui_message(text: str):
    from a2a.types import Message, Part, TextPart, Role

    return Message(
        messageId="ui-msg-1",
        role=Role.user,
        contextId="ctx-1",
        parts=[Part(root=TextPart(text=text))],
    )


def main() -> int:
    if not os.path.exists(NORA_TRANSFORM):
        print(f"SKIP: nora transform not found at {NORA_TRANSFORM}")
        return 0
    t = _load_transform()
    query = "What is the causal role of sanitation in poliovirus transmission?"

    # Default params
    out = t.transform(_ui_message(query), {})
    data_part = out.parts[0].root
    payload = data_part.data
    inner = payload["data"]

    failures = []
    if payload.get("kind") != "literature-theory-generation":
        failures.append(f"kind != literature-theory-generation: {payload.get('kind')}")
    if inner.get("theory_query") != query:
        failures.append("theory_query not preserved")
    if "paper_store" in inner:
        failures.append("UI path emitted paper_store (should never happen)")
    if inner.get("max_papers_to_retrieve") != 100:
        failures.append(f"default max_papers != 100: {inner.get('max_papers_to_retrieve')}")
    if inner.get("do_qualified_novelty_evaluation") is not True:
        failures.append("default novelty-eval flag wrong")

    # Param override
    out2 = t.transform(_ui_message(query), {"max_papers_to_retrieve": 25})
    inner2 = out2.parts[0].root.data["data"]
    if inner2.get("max_papers_to_retrieve") != 25:
        failures.append("param override for max_papers ignored")
    if "paper_store" in inner2:
        failures.append("UI path emitted paper_store under param override")

    print("transform emitted keys:", sorted(inner.keys()))
    if failures:
        print("CONTRACT FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("CONTRACT OK: UI TextPart -> literature-theory-generation DataPart, "
          "no paper_store, defaults + overrides correct. UI path is unaffected "
          "by the theorizer paper_store change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
