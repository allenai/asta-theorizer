"""Unit tests for the per-skill completion messages (src/asta/messages.py).

    .venv/bin/python tests/test_completion_messages.py
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asta.messages import theory_message  # noqa: E402


def _wf(num_theories):
    return SimpleNamespace(theory_store=SimpleNamespace(theories=[object()] * num_theories))


def test_theory_message_includes_credits_charged():
    assert theory_message(_wf(5), 511) == "Generated 5 theories using 511 credits:"
    assert theory_message(_wf(1), 1) == "Generated 1 theory using 1 credit:"
    return "credits in the sentence, singular/plural"


def test_theory_message_omits_credits_when_not_charged():
    assert theory_message(_wf(5), None) == "Generated 5 theories:"
    return "no credits clause without a charge"


def test_no_theories_message_omits_credits():
    assert theory_message(_wf(0), 3) == "No theories generated."
    assert theory_message(_wf(0), None) == "No theories generated."
    return "empty run shows no credits"


TESTS = [
    test_theory_message_includes_credits_charged,
    test_theory_message_omits_credits_when_not_charged,
    test_no_theories_message_omits_credits,
]


def main():
    failures = 0
    for fn in TESTS:
        try:
            print(f"PASS  {fn.__name__}: {fn()}", flush=True)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {fn.__name__}: {e!r}", flush=True)
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
