#!/usr/bin/env python3
"""E2E A/B test: asta-papers primary fetch path recovers ADDITIONAL papers.

Runs the same 3 papers through PaperStore._add_paper twice -- once with
USE_ASTA_PAPERS off (legacy S2-URL + Mistral OCR only) and once on -- and
asserts the on-run recovers strictly more full-text than the legacy baseline.

The 3 papers:
  * 2 "additional" targets: biomedical papers whose S2 record has NO
    openAccessPdf / arXiv / ACL id, so the legacy path yields zero URLs and
    returns metadata-only. asta-papers recovers them via the PMC JATS channel.
  * 1 control: an arXiv paper both paths recover -- proves the flag doesn't
    regress papers the legacy path already handled.

Real network: S2 lookup + asta-papers (PMC efetch) + Mistral OCR (control).
Keys load from src/api_keys.donotcommit.json (mistral) and .env (S2).

Run:
    PYTHONPATH=src .venv/bin/python tests/paperstore_asta_papers_ab.py
Exit code 0 iff asta-papers recovers strictly more full-text than legacy.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

os.environ["MISTRAL_API_KEY"] = json.load(open(os.path.join(REPO, "src/api_keys.donotcommit.json")))["mistral"]
for _line in open(os.path.join(REPO, ".env")):
    if _line.startswith("SEMANTIC_SCHOLAR_API_KEY="):
        os.environ["SEMANTIC_SCHOLAR_API_KEY"] = _line.split("=", 1)[1].strip()

from PaperStore import PaperStore  # noqa: E402

# (label, kwargs for _add_paper, expected_legacy_full_text)
# The two targets have a PMCID but no openAccessPdf/arXiv/ACL in S2 -> legacy
# returns metadata-only; asta-papers serves them via pmc-efetch JATS.
PAPERS = [
    ("target:proteomic-kras (PMC-only)", dict(corpus_id="282800036"), False),
    ("target:kras-g12c-drug-therapy (PMC-only)", dict(corpus_id="284717374"), False),
    ("control:attention-is-all-you-need (arXiv)", dict(paper_title="Attention Is All You Need"), True),
]


def fetch(flag, kw):
    os.environ["USE_ASTA_PAPERS"] = "true" if flag else "false"
    ps = PaperStore()
    key = ps._add_paper(**kw)
    rec = ps.get_paper_by_key(key) if key else None
    if not rec:
        return {"status": "NO_RECORD", "fetch_channel": None, "chars": 0}
    md = rec.get("paper_markdown") or ""
    return {"status": rec.get("status"), "fetch_channel": rec.get("fetch_channel"), "chars": len(md)}


def main():
    rows = []
    for label, kw, expect_legacy in PAPERS:
        off = fetch(False, kw)
        on = fetch(True, kw)
        rows.append((label, expect_legacy, off, on))

    print("\n" + "=" * 100)
    print(f"{'paper':<44} {'legacy(off)':<22} {'asta-papers(on)':<26}")
    print("-" * 100)
    off_ft = on_ft = 0
    additional = []
    regressions = []
    for label, expect_legacy, off, on in rows:
        off_full = off["status"] == "full-text"
        on_full = on["status"] == "full-text"
        off_ft += off_full
        on_ft += on_full
        if on_full and not off_full:
            additional.append(label)
        if off_full and not on_full:
            regressions.append(label)
        print(f"{label:<44} "
              f"{off['status']:<11}{off['chars']:>7}c   "
              f"{(on['fetch_channel'] or '-'):<13}{on['status']:<11}{on['chars']:>7}c")
    print("=" * 100)
    print(f"full-text recovered: legacy(off)={off_ft}/{len(PAPERS)}  asta-papers(on)={on_ft}/{len(PAPERS)}")
    print(f"additional papers recovered by asta-papers: {additional}")
    if regressions:
        print(f"REGRESSIONS (off recovered, on did not): {regressions}")

    # Assertions: on must recover strictly more, with no regressions, and the
    # two PMC-only targets must flip metadata-only -> full-text via a PMC channel.
    assert not regressions, f"asta-papers regressed papers the legacy path recovered: {regressions}"
    assert on_ft > off_ft, f"expected asta-papers to recover MORE full-text; got on={on_ft} off={off_ft}"
    target_flips = [lbl for (lbl, exp, off, on) in rows
                    if lbl.startswith("target:") and off["status"] != "full-text" and on["status"] == "full-text"]
    assert len(target_flips) >= 2, f"expected both PMC-only targets to flip to full-text; flipped: {target_flips}"
    print("\nPASS: asta-papers fetches additional full-text papers the legacy path misses, no regressions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
