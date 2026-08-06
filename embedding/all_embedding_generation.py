"""
all_embedding_generation.py

Bulk-registers all persons found in the /images folder, using a
thread pool to process multiple people concurrently (I/O-bound work:
DB queries/inserts, file writes) while the DeepFace model itself is
warmed up sequentially first to avoid a race in its internal model
cache.

Connects to PostgreSQL (IdentityDB) using credentials from a local
.env file (embedding/.env).
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, MetaData, Table, Column, Integer, String, insert, select
from sqlalchemy.exc import IntegrityError
from pgvector.sqlalchemy import Vector

from embedding_pipeline import FaceEmbeddingGenerator, VectorStoreWriter
from embedding_tracing import EmbeddingTracer, DetectedFaceSaver


EMBEDDING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EMBEDDING_DIR)
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
MAX_PARALLEL_WORKERS = 3   # kept modest — CPU-bound inference + shared model cache


load_dotenv(dotenv_path=os.path.join(EMBEDDING_DIR, ".env"))


class DatabaseConfig:
    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", "5432"))
        self.database_name = os.getenv("DB_NAME", "IdentityDB")
        self.username = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "")

    def build_connection_url(self):
        return URL.create(
            drivername="postgresql+psycopg2",
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database_name,
        )


_engine = create_engine(DatabaseConfig().build_connection_url(), echo=False)
_metadata = MetaData()

identity_embedding_table = Table(
    "identity_embedding_table",
    _metadata,
    Column("person_unique_id", Integer, primary_key=True),
    Column("person_name", String(150), nullable=False, unique=True),
    Column("facial_data", Vector(512), nullable=False),
)


class ImageFolderScanner:
    def __init__(self, images_dir: str = IMAGES_DIR):
        self.images_dir = images_dir

    def scan(self) -> list:
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Images folder not found at: {self.images_dir}")

        image_entries = []
        for file_name in sorted(os.listdir(self.images_dir)):
            file_path = os.path.join(self.images_dir, file_name)
            if not os.path.isfile(file_path):
                continue

            name_part, extension = os.path.splitext(file_name)
            if extension.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            image_entries.append((name_part, file_path))

        return image_entries


class PersonExistenceChecker:
    """
    Thread-safe: SQLAlchemy Engine manages its own connection pool,
    so each thread checking out its own connection via engine.connect()
    is safe without extra locking.
    """

    def __init__(self, engine):
        self.engine = engine

    def exists(self, person_name: str) -> bool:
        select_statement = select(identity_embedding_table.c.person_unique_id).where(
            identity_embedding_table.c.person_name == person_name
        )
        with self.engine.connect() as connection:
            result = connection.execute(select_statement)
            return result.fetchone() is not None


class BulkPersonRegistrar:
    """
    Orchestrates bulk registration with multithreading: several
    people's images are processed concurrently (I/O-bound: DB checks,
    DB inserts, file writes). The first image is processed sequentially
    as a warm-up so DeepFace's internal model cache is loaded once
    before parallel threads start (avoids a race on first load).
    """

    def __init__(self):
        self.engine = _engine
        self.scanner = ImageFolderScanner()
        self.existence_checker = PersonExistenceChecker(self.engine)

        # Stateless / safe to share across threads
        self.embedding_generator = FaceEmbeddingGenerator()
        self.vector_store_writer = VectorStoreWriter()
        self.face_saver = DetectedFaceSaver()

        # Synchronization
        self._tracing_lock = threading.Lock()   # protects tracing.json writes
        self._counts_lock = threading.Lock()    # protects shared counters

        self.registered_count = 0
        self.skipped_count = 0
        self.failed_count = 0

    def run(self) -> None:
        image_entries = self.scanner.scan()

        if not image_entries:
            print(f"[WARNING] No supported images found in: {self.scanner.images_dir}")
            return

        print(f"[INFO] Found {len(image_entries)} image(s) in {self.scanner.images_dir}")
        start_time = time.perf_counter()

        # Warm-up: process the first image sequentially so DeepFace loads
        # and caches RetinaFace + FaceNet512 once, avoiding a race
        # condition if multiple threads triggered the first load at once.
        first_name, first_path = image_entries[0]
        print(f"[INFO] Warming up models with '{first_name}' before parallel processing...")
        self._process_single_person(first_name, first_path)

        remaining_entries = image_entries[1:]
        if remaining_entries:
            print(f"[INFO] Processing remaining {len(remaining_entries)} image(s) with "
                  f"{MAX_PARALLEL_WORKERS} parallel worker(s)...")
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
                futures = {
                    executor.submit(self._process_single_person, name, path): name
                    for name, path in remaining_entries
                }
                for future in as_completed(futures):
                    future.result()  # re-raises unexpected exceptions (handled ones are already caught)

        elapsed_seconds = time.perf_counter() - start_time

        print(
            f"\n[SUMMARY] Registered: {self.registered_count} | "
            f"Skipped (already exists): {self.skipped_count} | "
            f"Failed: {self.failed_count} | "
            f"Elapsed: {elapsed_seconds:.2f}s"
        )

    def _process_single_person(self, person_name: str, image_path: str) -> None:
        print(f"\n[INFO] Processing '{person_name}' -> {image_path}")

        # A dedicated tracer instance per task — NOT shared across
        # threads, since it holds mutable per-run state internally.
        tracer = EmbeddingTracer(lock=self._tracing_lock)
        tracer.start_run(image_path)

        try:
            tracer.start_step("check_person_exists")
            already_exists = self.existence_checker.exists(person_name)
            tracer.end_step(status="SUCCESS", details=f"person_exists={already_exists}")

            if already_exists:
                print(f"[WARNING] '{person_name}' already exists in the database. Skipping.")
                self._increment(skipped=True)
                tracer.end_run(status="SKIPPED", error_message="person_name already exists in database")
                return

            tracer.start_step("detect_and_save_face_retinaface")
            detected_face_path = self.face_saver.save_detected_face(image_path, person_name=person_name)
            tracer.end_step(status="SUCCESS", details=f"Detected face saved to {detected_face_path}")

            tracer.start_step("generate_embedding_facenet512")
            facial_embedding = self.embedding_generator.generate_embedding(image_path)
            tracer.end_step(status="SUCCESS", details=f"Generated embedding of length {len(facial_embedding)}")

            tracer.start_step("store_embedding_json")
            saved_json_path = self.vector_store_writer.save_embedding(
                file_path=image_path,
                file_name=os.path.basename(image_path),
                embedding_vector=facial_embedding,
            )
            tracer.end_step(status="SUCCESS", details=f"Saved to {saved_json_path}")

            tracer.start_step("insert_database_record")
            self._insert_person(person_name, facial_embedding)
            tracer.end_step(status="SUCCESS", details=f"Inserted '{person_name}' into identity_embedding_table")

            print(f"[SUCCESS] '{person_name}' registered successfully.")
            self._increment(registered=True)
            tracer.end_run(status="SUCCESS")

        except (FileNotFoundError, ValueError) as error:
            print(f"[ERROR] Failed to process '{person_name}': {error}")
            tracer.end_step(status="FAILED", error_message=str(error))
            tracer.end_run(status="FAILED", error_message=str(error))
            self._increment(failed=True)

        except IntegrityError as error:
            print(f"[ERROR] Database error while inserting '{person_name}': {error.orig}")
            tracer.end_step(status="FAILED", error_message=str(error.orig))
            tracer.end_run(status="FAILED", error_message=str(error.orig))
            self._increment(failed=True)

    def _increment(self, registered: bool = False, skipped: bool = False, failed: bool = False) -> None:
        with self._counts_lock:
            if registered:
                self.registered_count += 1
            if skipped:
                self.skipped_count += 1
            if failed:
                self.failed_count += 1

    def _insert_person(self, person_name: str, facial_embedding: list) -> None:
        insert_statement = insert(identity_embedding_table).values(
            person_name=person_name,
            facial_data=facial_embedding,
        )
        with self.engine.begin() as connection:
            connection.execute(insert_statement)


def main():
    registrar = BulkPersonRegistrar()
    registrar.run()


if __name__ == "__main__":
    main()