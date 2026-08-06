"""
embedding_generation.py

Generates a 512D face embedding (RetinaFace + FaceNet512) from an
image file path, for use by database_admin/main.py's Add and Update
operations.

Relies on embedding/weights_loading.py to point DeepFace at the
project's local /models weights before deepface is imported.
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


class AdminFaceEmbeddingGenerator:
    """
    Detects and aligns a face using RetinaFace, then generates a
    512-dimensional embedding using FaceNet512, from an image file path.
    """

    def __init__(self, detector_backend: str = DETECTOR_BACKEND,
                 recognition_model: str = RECOGNITION_MODEL):
        self.detector_backend = detector_backend
        self.recognition_model = recognition_model

    def generate_embedding(self, image_path: str) -> list:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        representation_result = DeepFace.represent(
            img_path=image_path,
            model_name=self.recognition_model,
            detector_backend=self.detector_backend,
            enforce_detection=True,
            align=True,
        )

        if not representation_result:
            raise ValueError(f"No face detected in image: {image_path}")

        embedding_vector = representation_result[0]["embedding"]

        if len(embedding_vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Unexpected embedding size: got {len(embedding_vector)}, expected {EMBEDDING_DIMENSIONS}."
            )

        return embedding_vector