# TestPackaging.py
# The package is a library only: it installs no commands.  The `theorize` command line ended with
# 0.1.5; the HTML export runs as `python -m theorizer.Export`.
#
# Runs offline.
#
#     python tests/TestPackaging.py

import os
import sys
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def test_no_console_commands():
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as f:
        project = tomllib.load(f)["project"]
    assert ("scripts" not in project) and ("gui-scripts" not in project), "pyproject.toml declares commands: " + str(project.get("scripts"))


def test_the_export_runs_as_a_module():
    import theorizer.Export as export
    assert callable(export.main) and callable(export.export_html)


def main():
    print("TestPackaging.py")
    tests = [test_no_console_commands, test_the_export_runs_as_a_module]
    for test in tests:
        test()
        print("  " + test.__name__ + ": OK")
    print("TestPackaging.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
