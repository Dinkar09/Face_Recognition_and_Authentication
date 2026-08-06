"""
embedding_tracing.py

Traces the embedding_generation pipeline and stores execution history
in embedding/tracing/tracing.json. Each run (one image processed) is
appended as a separate record, so tracing.json accumulates a full
history across multiple executions.

Also provides DetectedFaceSaver, which saves the face detected by
RetinaFace as a PNG image inside embedding/tracing/face_detected/,
useful for visually verifying what the detector locked onto for
each person.
"""

import os
import json
import time
import uuid
import threading
from datetime import datetime, timezone

from PIL import Image

EMBEDDING_DIR = os.path.dirname(os.path.abspath(__file__))
TRACING_DIR = os.path.join(EMBEDDING_DIR, "tracing")
TRACING_FILE_PATH = os.path.join(TRACING_DIR, "tracing.json")

FACE_DETECTED_DIR = os.path.join(TRACING_DIR, "face_detected")


class PipelineStepTrace:
    """Represents a single traced step within a pipeline run."""

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.status = "PENDING"
        self.start_time = None
        self.end_time = None
        self.duration_seconds = None
        self.details = None
        self.error_message = None

    def start(self) -> None:
        self.start_time = time.time()

    def finish(self, status: str, details: str = None, error_message: str = None) -> None:
        self.end_time = time.time()
        self.duration_seconds = round(self.end_time - self.start_time, 4)
        self.status = status
        self.details = details
        self.error_message = error_message

    def to_dict(self) -> dict:
        return {
            "step_name": self.step_name,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "details": self.details,
            "error_message": self.error_message,
        }


class EmbeddingTracer:
    """
    Traces a full run of the embedding generation pipeline
    (one image/person, start to finish) and persists it to
    embedding/tracing/tracing.json.
    """

    def __init__(self, tracing_file_path: str = TRACING_FILE_PATH, lock: threading.Lock = None):
        self.tracing_file_path = tracing_file_path
        os.makedirs(os.path.dirname(self.tracing_file_path), exist_ok=True)
        self.lock = lock   # optional external lock for thread-safe writes

        self.run_id = None
        self.image_path = None
        self.run_start_time = None
        self.run_end_time = None
        self.steps = []
        self.overall_status = "PENDING"
        self.overall_error_message = None
        self._active_step = None

    # ------------------------- Run lifecycle ------------------------- #

    def start_run(self, image_path: str) -> None:
        self.run_id = str(uuid.uuid4())
        self.image_path = image_path
        self.run_start_time = time.time()
        self.steps = []
        self.overall_status = "IN_PROGRESS"
        self.overall_error_message = None

    def end_run(self, status: str, error_message: str = None) -> None:
        self.run_end_time = time.time()
        self.overall_status = status
        self.overall_error_message = error_message
        self._persist_run()

    # ------------------------- Step lifecycle ------------------------- #

    def start_step(self, step_name: str) -> None:
        step = PipelineStepTrace(step_name)
        step.start()
        self._active_step = step
        self.steps.append(step)

    def end_step(self, status: str = "SUCCESS", details: str = None, error_message: str = None) -> None:
        if self._active_step is None:
            return
        self._active_step.finish(status=status, details=details, error_message=error_message)
        self._active_step = None

    # ------------------------------ Persist ---------------------------- #

    def _build_run_record(self) -> dict:
        total_duration = None
        if self.run_start_time and self.run_end_time:
            total_duration = round(self.run_end_time - self.run_start_time, 4)

        return {
            "run_id": self.run_id,
            "image_path": self.image_path,
            "started_at": self._to_iso(self.run_start_time),
            "ended_at": self._to_iso(self.run_end_time),
            "total_duration_seconds": total_duration,
            "overall_status": self.overall_status,
            "overall_error_message": self.overall_error_message,
            "steps": [step.to_dict() for step in self.steps],
        }

    def _persist_run(self) -> None:
        if self.lock is not None:
            with self.lock:
                self._write_run_record()
        else:
            self._write_run_record()

    def _write_run_record(self) -> None:
        existing_runs = self._load_existing_runs()
        existing_runs.append(self._build_run_record())
        with open(self.tracing_file_path, "w") as tracing_file:
            json.dump(existing_runs, tracing_file, indent=4)

    def _load_existing_runs(self) -> list:
        if not os.path.isfile(self.tracing_file_path):
            return []

        try:
            with open(self.tracing_file_path, "r") as tracing_file:
                data = json.load(tracing_file)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    @staticmethod
    def _to_iso(unix_timestamp: float) -> str:
        if unix_timestamp is None:
            return None
        return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc).isoformat()


class DetectedFaceSaver:
    """
    Uses RetinaFace (via DeepFace.extract_faces) to detect and crop
    the face from an input image, then saves it as a PNG file inside
    embedding/tracing/face_detected/. Useful for visually verifying
    what the detector locked onto for each person.
    """

    def __init__(self, output_dir: str = FACE_DETECTED_DIR, detector_backend: str = "retinaface"):
        self.output_dir = output_dir
        self.detector_backend = detector_backend
        os.makedirs(self.output_dir, exist_ok=True)

    def save_detected_face(self, image_path: str, person_name: str = None) -> str:
        """
        Detects the face in image_path using RetinaFace and saves the
        cropped face as a PNG file named face_detected_<person_name>.png.

        If person_name is not provided, falls back to the input
        image's filename (without extension).

        Returns:
            str: Path to the saved PNG file.
        """
        from deepface import DeepFace  # local import: deepface must be configured via weights_loading first

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        extracted_faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=self.detector_backend,
            align=True,
            normalize_face=False,  # keep pixel values in 0-255 range for saving as PNG
        )

        if not extracted_faces:
            raise ValueError(f"No face detected in image: {image_path}")

        face_array = extracted_faces[0]["face"].astype("uint8")
        face_image = Image.fromarray(face_array)

        name_for_file = person_name or os.path.splitext(os.path.basename(image_path))[0]
        output_file_name = f"face_detected_{name_for_file}.png"
        output_path = os.path.join(self.output_dir, output_file_name)

        face_image.save(output_path)
        return output_path