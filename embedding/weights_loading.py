"""
weights_loading.py

Ensures DeepFace uses the already-downloaded weight files from the
project's /models folder instead of re-downloading them into
DeepFace's default cache directory (~/.deepface/weights).

Must be imported BEFORE `from deepface import DeepFace` anywhere in
the project, since DeepFace reads the DEEPFACE_HOME environment
variable at import time.
"""

import os
import shutil

class LocalWeightsConfigurator:
    """
    Points DeepFace's weight lookup to the project's local /models
    folder by setting DEEPFACE_HOME and copying existing .h5 files
    into the folder structure DeepFace expects
    (<DEEPFACE_HOME>/.deepface/weights/).
    """

    REQUIRED_WEIGHT_FILES = ["facenet512_weights.h5", "retinaface.h5"]

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.models_dir = os.path.join(project_root, "models")
        self.deepface_home = self.models_dir
        self.deepface_weights_dir = os.path.join(
            self.deepface_home, ".deepface", "weights"
        )

    def configure(self) -> None:
        os.makedirs(self.deepface_weights_dir, exist_ok=True)
        self._copy_existing_weights_if_needed()
        os.environ["DEEPFACE_HOME"] = self.deepface_home

    def _copy_existing_weights_if_needed(self) -> None:
        for weight_file_name in self.REQUIRED_WEIGHT_FILES:
            source_path = os.path.join(self.models_dir, weight_file_name)
            destination_path = os.path.join(self.deepface_weights_dir, weight_file_name)

            if os.path.isfile(destination_path):
                continue  # Already in place, skip.

            if os.path.isfile(source_path):
                shutil.copyfile(source_path, destination_path)
                print(f"[INFO] Linked local weight file: {weight_file_name}")
            else:
                print(
                    f"[WARNING] {weight_file_name} not found in {self.models_dir}. "
                    "DeepFace will download it automatically instead."
                )


def configure_local_weights() -> None:
    """
    Convenience function: locates the project root and configures
    DeepFace to use local weights. Call this before importing deepface.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LocalWeightsConfigurator(project_root).configure()


# Run configuration immediately on import, so any module that does
# `import weights_loading` before `from deepface import DeepFace`
# gets DEEPFACE_HOME set correctly first.
configure_local_weights()