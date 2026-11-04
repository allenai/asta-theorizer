# Persistence.py
# Persistence interface with two implementations:
# - FilePersistence (default, file-based)
# - PostgresPersistence (uses psycopg2-binary, selected when DATABASE_URL is set)

import os
import json
import base64
from abc import ABC, abstractmethod


class PersistenceBackend(ABC):
    @abstractmethod
    def save_workflow(self, generation_id: str, data: dict) -> None: ...

    @abstractmethod
    def get_workflow(self, generation_id: str) -> dict | None: ...

    @abstractmethod
    def save_theorystore(self, generation_id: str, data: dict) -> None: ...

    @abstractmethod
    def get_theorystore(self, generation_id: str) -> dict | None: ...

    @abstractmethod
    def save_paperstore(self, generation_id: str, data: dict) -> None: ...

    @abstractmethod
    def get_paperstore(self, generation_id: str): ...

    @abstractmethod
    def save_ocr_cache(self, url_key: str, ocr_json: dict, markdown: str) -> None: ...

    @abstractmethod
    def load_ocr_cache_markdown(self, url_key: str) -> str | None: ...

    @abstractmethod
    def get_kv(self, key: str) -> dict | None: ...

    @abstractmethod
    def set_kv(self, key: str, value: dict) -> None: ...


# ---------------------------------------------------------------------------
#  File-based persistence (default)
# ---------------------------------------------------------------------------

# Project root is one level above this file (src/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class FilePersistence(PersistenceBackend):
    def __init__(self, base_path: str | None = None):
        if base_path is None:
            base_path = os.path.join(_PROJECT_ROOT, "data")
        self.path_workflows = os.path.join(base_path, "workflows")
        self.path_theorystores = os.path.join(base_path, "theorystores")
        self.path_paperstores = os.path.join(base_path, "paperstores")
        self.path_ocr_cache = os.path.join(_PROJECT_ROOT, "ocr-cache")
        self.base_path = base_path
        os.makedirs(self.path_workflows, exist_ok=True)
        os.makedirs(self.path_theorystores, exist_ok=True)
        os.makedirs(self.path_paperstores, exist_ok=True)
        os.makedirs(self.path_ocr_cache, exist_ok=True)

    def save_workflow(self, generation_id: str, data: dict) -> None:
        filename = os.path.join(self.path_workflows, f"workflow_id_{generation_id}.json")
        print(f"FilePersistence.save_workflow(): Saving workflow to file: {filename}")
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def get_workflow(self, generation_id: str) -> dict | None:
        filename = os.path.join(self.path_workflows, f"workflow_id_{generation_id}.json")
        if not os.path.exists(filename):
            return None
        with open(filename, "r") as f:
            return json.load(f)

    def save_theorystore(self, generation_id: str, data: dict) -> None:
        filename = os.path.join(self.path_theorystores, f"theorystore_{generation_id}.json")
        print(f"FilePersistence.save_theorystore(): Saving theorystore to file: {filename}")
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def get_theorystore(self, generation_id: str) -> dict | None:
        filename = os.path.join(self.path_theorystores, f"theorystore_{generation_id}.json")
        if not os.path.exists(filename):
            return None
        with open(filename, "r") as f:
            return json.load(f)

    def save_paperstore(self, generation_id: str, data: dict) -> None:
        filename = os.path.join(self.path_paperstores, f"paperstore_{generation_id}.json")
        print(f"FilePersistence.save_paperstore(): Saving paperstore to file: {filename}")
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def get_paperstore(self, generation_id: str):
        from PaperStore import PaperStore

        filename = os.path.join(self.path_paperstores, f"paperstore_{generation_id}.json")
        if not os.path.exists(filename):
            return None
        with open(filename, "r") as f:
            #return json.load(f)
            try:
                data = json.load(f)
                # Convert into a PaperStore object
                paperstore = PaperStore(deserialize_from_dict=data, generation_id=generation_id)
                return paperstore
            except Exception as e:
                print(f"Error loading paperstore from file {filename}: {e}")
                return None

    def save_ocr_cache(self, url_key: str, ocr_json: dict, markdown: str) -> None:
        sanitized = self._sanitize_url(url_key)
        cache_dir = os.path.join(self.path_ocr_cache, sanitized)
        os.makedirs(cache_dir, exist_ok=True)
        # Save JSON
        json_path = os.path.join(cache_dir, "ocr_response.json")
        with open(json_path, "w") as f:
            json.dump(ocr_json, f)
        # Save markdown
        md_path = os.path.join(cache_dir, "ocr_response.md")
        with open(md_path, "w") as f:
            f.write(markdown)
        # Save base64-encoded images from OCR response
        for page in ocr_json.get("pages", []):
            for img in page.get("images", []):
                try:
                    img_id = img["id"]
                    data_uri = img["image_base64"]
                    _, encoded = data_uri.split(",", 1)
                    img_data = base64.b64decode(encoded)
                    image_path = os.path.join(cache_dir, img_id)
                    with open(image_path, "wb") as f:
                        f.write(img_data)
                except Exception as e:
                    print(f"Error saving image {img.get('id', 'unknown')}: {e}")

    def load_ocr_cache_markdown(self, url_key: str) -> str | None:
        sanitized = self._sanitize_url(url_key)
        md_path = os.path.join(self.path_ocr_cache, sanitized, "ocr_response.md")
        if not os.path.exists(md_path):
            return None
        with open(md_path, "r") as f:
            return f.read()

    def get_kv(self, key: str) -> dict | None:
        filename = os.path.join(self.base_path, f"{key}.json")
        if not os.path.exists(filename):
            return None
        with open(filename, "r") as f:
            return json.load(f)

    def set_kv(self, key: str, value: dict) -> None:
        filename = os.path.join(self.base_path, f"{key}.json")
        with open(filename, "w") as f:
            json.dump(value, f)

    @staticmethod
    def _sanitize_url(url: str) -> str:
        if url.startswith("http://"):
            url = url[7:]
        elif url.startswith("https://"):
            url = url[8:]
        sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in url)
        return sanitized.strip('_')


# ---------------------------------------------------------------------------
#  Postgres-based persistence (selected when DATABASE_URL is set)
# ---------------------------------------------------------------------------

class PostgresPersistence(PersistenceBackend):
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._create_tables()

    def _connect(self):
        import psycopg2
        return psycopg2.connect(self.database_url)

    def _create_tables(self):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS workflows (
                        generation_id TEXT PRIMARY KEY,
                        data JSONB NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS theorystores (
                        generation_id TEXT PRIMARY KEY,
                        data JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS paperstores (
                        generation_id TEXT PRIMARY KEY,
                        data JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ocr_cache (
                        url_key TEXT PRIMARY KEY,
                        ocr_json JSONB NOT NULL,
                        markdown TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS kv_store (
                        key TEXT PRIMARY KEY,
                        value JSONB NOT NULL
                    )
                """)
            conn.commit()
        finally:
            conn.close()

    def save_workflow(self, generation_id: str, data: dict) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                # Upsert: insert or update if generation_id already exists
                cur.execute(
                    """INSERT INTO workflows (generation_id, data, updated_at)
                       VALUES (%s, %s, NOW())
                       ON CONFLICT (generation_id) DO UPDATE
                       SET data = EXCLUDED.data, updated_at = NOW()""",
                    (generation_id, json.dumps(data))
                )
            conn.commit()
        finally:
            conn.close()

    def get_workflow(self, generation_id: str) -> dict | None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT data FROM workflows WHERE generation_id = %s",
                    (generation_id,)
                )
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            conn.close()

    def save_theorystore(self, generation_id: str, data: dict) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                # Upsert: insert or update if generation_id already exists
                cur.execute(
                    """INSERT INTO theorystores (generation_id, data, created_at)
                       VALUES (%s, %s, NOW())
                       ON CONFLICT (generation_id) DO UPDATE
                       SET data = EXCLUDED.data""",
                    (generation_id, json.dumps(data))
                )
            conn.commit()
        finally:
            conn.close()

    def get_theorystore(self, generation_id: str) -> dict | None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT data FROM theorystores WHERE generation_id = %s",
                    (generation_id,)
                )
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            conn.close()

    def save_paperstore(self, generation_id: str, data: dict) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO paperstores (generation_id, data, created_at)
                       VALUES (%s, %s, NOW())
                       ON CONFLICT (generation_id) DO UPDATE
                       SET data = EXCLUDED.data""",
                    (generation_id, json.dumps(data))
                )
            conn.commit()
        finally:
            conn.close()

    def get_paperstore(self, generation_id: str):
        from PaperStore import PaperStore

        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT data FROM paperstores WHERE generation_id = %s",
                    (generation_id,)
                )
                row = cur.fetchone()
                #return row[0] if row else None
                data = row[0] if row else None
                if (data is None):
                    return None
                paperstore = PaperStore(deserialize_from_dict=data, generation_id=generation_id)
                return paperstore

        finally:
            conn.close()

    def save_ocr_cache(self, url_key: str, ocr_json: dict, markdown: str) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                # Upsert: insert or update if url_key already exists
                cur.execute(
                    """INSERT INTO ocr_cache (url_key, ocr_json, markdown)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (url_key) DO UPDATE
                       SET ocr_json = EXCLUDED.ocr_json, markdown = EXCLUDED.markdown""",
                    (url_key, json.dumps(ocr_json), markdown)
                )
            conn.commit()
        finally:
            conn.close()

    def load_ocr_cache_markdown(self, url_key: str) -> str | None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT markdown FROM ocr_cache WHERE url_key = %s",
                    (url_key,)
                )
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            conn.close()

    def get_kv(self, key: str) -> dict | None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT value FROM kv_store WHERE key = %s",
                    (key,)
                )
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            conn.close()

    def set_kv(self, key: str, value: dict) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                # Upsert: insert or update if key already exists
                cur.execute(
                    """INSERT INTO kv_store (key, value)
                       VALUES (%s, %s)
                       ON CONFLICT (key) DO UPDATE
                       SET value = EXCLUDED.value""",
                    (key, json.dumps(value))
                )
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
#  Factory + singleton
# ---------------------------------------------------------------------------

def create_persistence_backend() -> PersistenceBackend:
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        print("persistence: Using PostgresPersistence (DATABASE_URL is set)")
        return PostgresPersistence(database_url)
    print("persistence: Using FilePersistence (no DATABASE_URL)")
    return FilePersistence()


_store: PersistenceBackend | None = None


def store() -> PersistenceBackend:
    global _store
    if _store is None:
        _store = create_persistence_backend()
    return _store
