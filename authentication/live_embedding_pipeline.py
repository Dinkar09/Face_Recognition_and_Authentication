"""
live_embedding_pipeline.py

Lightweight embedding pipeline for live video authentication.
Generates a 512D FaceNet512 embedding directly from an in-memory
video frame (numpy.ndarray) using RetinaFace for detection/alignment.

Unlike embedding/embedding_pipeline.py (used for registration), this
version does NOT write to vector_store/ or tracing.json — it only
returns the embedding in memory, since live authentication runs
continuously and per-frame disk I/O would add unnecessary overhead.
Logging of authentication attempts is handled separately by log.py.

Relies on embedding/weights_loading.py to point DeepFace at local
weights before deepface is imported.
"""

import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
_EMBEDDING_DIR = os.path.join(_PROJECT_ROOT, "embedding")

if _EMBEDDING_DIR not in sys.path:
    sys.path.insert(0, _EMBEDDING_DIR)

import weights_loading  # noqa: F401  (must be imported before deepface)
from deepface import DeepFace


DETECTOR_BACKEND = "retinaface"
RECOGNITION_MODEL = "Facenet512"
EMBEDDING_DIMENSIONS = 512


class LiveFaceEmbeddingGenerator:
    """
    Generates a 512D embedding from an in-memory video frame using
    RetinaFace (detection & alignment) + FaceNet512 (recognition).
    """

    def __init__(self, detector_backend: str = DETECTOR_BACKEND,
                 recognition_model: str = RECOGNITION_MODEL):
        self.detector_backend = detector_backend
        self.recognition_model = recognition_model

    def generate_embedding(self, frame) -> list:
        """
        Args:
            frame: numpy.ndarray (BGR frame from OpenCV VideoCapture)

        Returns:
            list[float]: 512-dimensional embedding vector.
        """
        representation_result = DeepFace.represent(
            img_path=frame,
            model_name=self.recognition_model,
            detector_backend=self.detector_backend,
            enforce_detection=True,
            align=True,
        )

        if not representation_result:
            raise ValueError("No face detected in the captured frame.")

        first_face = representation_result[0]
        embedding_vector = first_face["embedding"]

        if len(embedding_vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Unexpected embedding size: got {len(embedding_vector)}, expected {EMBEDDING_DIMENSIONS}."
            )

        return embedding_vector