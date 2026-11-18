"""Export a theorystore to HTML.

The exporter is part of the library now (theorizer/Export.py), so it also works from a pip install,
as `python -m theorizer.Export`.  This script stays so the command below keeps working from a clone.

    python export_theories_html.py --project my-run                  # my-run/theorystore.json -> my-run/html/
    python export_theories_html.py --store my-run/theorystore.json --paperstore my-run/paperstore.json --out my-run/html
    python export_theories_html.py --store my-run/theorystore.json --out my-run-html --limit 20   # a quick subset
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from theorizer.Export import main  # noqa: E402

if (__name__ == "__main__"):
    sys.exit(main())
