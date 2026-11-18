# RunAllTests.py
# Run every test.  All are offline.
#
#     python tests/RunAllTests.py

import os
import subprocess
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_FILES = [
    "TestLegacy.py",
    "TestConfig.py",
    "TestProject.py",
    "TestParallel.py",
    "TestPaperImport.py",
    "TestPipeline.py",
    "TestLLMTimeout.py",
    "TestGenerationTimeout.py",
    "TestModels.py",
    "TestNoPaperSearch.py",
    "TestEvidenceChain.py",
    "TestSymbolic.py",
    "TestPromptCache.py",
    "TestPdfRoute.py",
    "TestPdfUrls.py",
    "TestMaxTokens.py",
    "TestNoveltyStore.py",
    "TestNoveltyExport.py",
    "TestEvidenceSeed.py",
    "TestResume.py",
    "TestPersistence.py",
    "TestWritesInProject.py",
    "TestPackaging.py",
    "test_mistral_ocr_error_logging.py",
]


def main():
    failed = []
    for filename in TEST_FILES:
        path = os.path.join(TESTS_DIR, filename)
        result = subprocess.run([sys.executable, path])
        if (result.returncode != 0):
            failed.append(filename)
        print("")

    if (len(failed) > 0):
        print("RunAllTests.py: FAILED: " + ", ".join(failed))
        return 1

    print("RunAllTests.py: All " + str(len(TEST_FILES)) + " test file(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
