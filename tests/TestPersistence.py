# TestPersistence.py
# Where the engine keeps its caches.  Postgres is chosen by THEORIZER_DATABASE_URL only: a generic
# DATABASE_URL that belongs to something else in the environment leaves Theorizer on files.
#
# Runs offline.  No API keys, no database, no network.
#
#     python tests/TestPersistence.py

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th

import Persistence


class FakePostgres:
    def __init__(self, database_url):
        self.database_url = database_url


class environment:
    # Set (or, with None, remove) environment variables for the length of a block
    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.saved = {name: os.environ.get(name) for name in self.values}
        for name, value in self.values.items():
            if (value is None):
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            if (value is None):
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return False


def backend(directory):
    real = Persistence.PostgresPersistence
    Persistence.PostgresPersistence = FakePostgres
    th.Config.init(project_dir=os.path.join(directory, "run"), verbose=False)
    try:
        return Persistence.create_persistence_backend()
    finally:
        Persistence.PostgresPersistence = real
        th.Project.close()


def test_a_generic_database_url_is_ignored(directory):
    with environment(DATABASE_URL="postgres://someone-else/db", THEORIZER_DATABASE_URL=None):
        chosen = backend(directory)
    assert isinstance(chosen, Persistence.FilePersistence), "a generic DATABASE_URL switched Theorizer to Postgres: " + type(chosen).__name__


def test_theorizer_database_url_selects_postgres(directory):
    with environment(DATABASE_URL=None, THEORIZER_DATABASE_URL="postgres://theorizer/db"):
        chosen = backend(directory)
    assert isinstance(chosen, FakePostgres) and (chosen.database_url == "postgres://theorizer/db"), "THEORIZER_DATABASE_URL did not select Postgres: " + type(chosen).__name__


def test_no_url_means_files(directory):
    with environment(DATABASE_URL=None, THEORIZER_DATABASE_URL=None):
        chosen = backend(directory)
    assert isinstance(chosen, Persistence.FilePersistence), "with no URL the backend is " + type(chosen).__name__


def main():
    print("TestPersistence.py")
    tests = [
        test_a_generic_database_url_is_ignored,
        test_theorizer_database_url_selects_postgres,
        test_no_url_means_files,
    ]
    for test in tests:
        directory = tempfile.mkdtemp()
        try:
            test(directory)
            print("  " + test.__name__ + ": OK")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    print("TestPersistence.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
