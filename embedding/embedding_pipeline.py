"""
embedding_pipeline.py

Workflow:
Input Image (.png) --> RetinaFace (detect & align) --> FaceNet512 (512D embedding)
--> Store embedding as a unique JSON file per image in /vector_store

Each image gets its own JSON file (image1.json, image2.json, ...).
Every run is traced end-to-end and recorded in tracing.json via
embedding_tracing.EmbeddingTracer.

Relies on weights_loading.py to point DeepFace at local weights
before deepface is imported.
"""

import os
import sys
import json

import weights_loading  # noqa: F401  (must be imported before deepface)
from deepface import DeepFace

from embedding_tracing import EmbeddingTracer, DetectedFaceSaver


# ----------------------------- Configuration ----------------------------- #

DETECTOR_BACKEND = "retinaface"
RECOGNITION_MODEL = "Facenet512"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_store")


# ----------------------------- Core Classes ------------------------------ #

class FaceEmbeddingGenerator:
    """
    Detects and aligns a face using RetinaFace, then generates a
    512-dimensional embedding for that face using FaceNet512.
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

        first_face = representation_result[0]
        embedding_vector = first_face["embedding"]

        return embedding_vector


class VectorStoreWriter:
    """
    Handles saving a generated face embedding into a unique JSON file
    inside the vector_store folder.
    """

    def __init__(self, vector_store_dir: str = VECTOR_STORE_DIR):
        self.vector_store_dir = vector_store_dir
        os.makedirs(self.vector_store_dir, exist_ok=True)

    def save_embedding(self, file_path: str, file_name: str, embedding_vector: list) -> str:
        json_file_name = f"{os.path.splitext(file_name)[0]}.json"
        json_file_path = os.path.join(self.vector_store_dir, json_file_name)

        embedding_data = {
            "file_path": file_path,
            "file_name": file_name,
            "file_name_vector": embedding_vector,
        }

        with open(json_file_path, "w") as json_file:
            json.dump(embedding_data, json_file, indent=4)

        return json_file_path


class EmbeddingPipeline:
    """
    Orchestrates the full workflow:
    Image -> RetinaFace -> FaceNet512 -> JSON stored in vector_store
    Each run is traced via EmbeddingTracer and logged to tracing.json.
    """

    def __init__(self):
        self.embedding_generator = FaceEmbeddingGenerator()
        self.vector_store_writer = VectorStoreWriter()
        self.tracer = EmbeddingTracer()
        self.detected_face_saver = DetectedFaceSaver()

    def process_image(self, image_path: str) -> str:
        file_name = os.path.basename(image_path)
        self.tracer.start_run(image_path)

        try:
            print(f"[INFO] Processing image: {image_path}")

            self.tracer.start_step("detect_and_align_face_retinaface")
            print("[INFO] Step 1: Detecting & aligning face using RetinaFace...")
            detected_face_path = self.detected_face_saver.save_detected_face(image_path)
            self.tracer.end_step(status="SUCCESS", details=f"Detected face saved to {detected_face_path}")

            self.tracer.start_step("generate_embedding_facenet512")
            print("[INFO] Step 2: Generating 512D embedding using FaceNet512...")
            embedding_vector = self.embedding_generator.generate_embedding(image_path)
            self.tracer.end_step(
                status="SUCCESS",
                details=f"Generated embedding of length {len(embedding_vector)}",
            )

            self.tracer.start_step("store_embedding_json")
            print(f"[INFO] Step 3: Storing embedding JSON in: {self.vector_store_writer.vector_store_dir}")
            saved_json_path = self.vector_store_writer.save_embedding(
                file_path=image_path,
                file_name=file_name,
                embedding_vector=embedding_vector,
            )
            self.tracer.end_step(status="SUCCESS", details=f"Saved to {saved_json_path}")

            print(f"[SUCCESS] Embedding saved to: {saved_json_path}")
            self.tracer.end_run(status="SUCCESS")
            return saved_json_path

        except (FileNotFoundError, ValueError) as error:
            self.tracer.end_step(status="FAILED", error_message=str(error))
            self.tracer.end_run(status="FAILED", error_message=str(error))
            raise

        except Exception as error:
            self.tracer.end_step(status="FAILED", error_message=str(error))
            self.tracer.end_run(status="FAILED", error_message=str(error))
            raise


# ------------------------------- Entry Point ------------------------------ #

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = r"D:\Face Recognition and Authentication\images\Dinkar.png"

    pipeline = EmbeddingPipeline()

    try:
        pipeline.process_image(image_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"[ERROR] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()