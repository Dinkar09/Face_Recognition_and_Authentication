"""
retinaface_model_download.py

Downloads the pretrained RetinaFace weights file (used by DeepFace for
face detection) and stores it inside the /Model folder of the current project.

File type downloaded: .h5 (Keras HDF5 model weights)
"""

import os
import sys
import requests

# Official DeepFace model weights release (same source DeepFace uses internally)
RETINAFACE_WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/"
    "download/v1.0/retinaface.h5"
)

MODEL_DIR_NAME = "Model"
WEIGHTS_FILE_NAME = "retinaface.h5"


class ModelDownloader:
    """Handles downloading and saving pretrained model weight files."""

    def __init__(self, download_url: str, destination_folder: str, file_name: str):
        self.download_url = download_url
        self.destination_folder = destination_folder
        self.file_name = file_name
        self.destination_path = os.path.join(self.destination_folder, self.file_name)

    def ensure_destination_folder_exists(self) -> None:
        os.makedirs(self.destination_folder, exist_ok=True)

    def file_already_exists(self) -> bool:
        return os.path.isfile(self.destination_path)

    def download(self) -> None:
        if self.file_already_exists():
            print(f"[INFO] File already exists at: {self.destination_path}")
            print("[INFO] Skipping download.")
            return

        self.ensure_destination_folder_exists()

        print(f"[INFO] Downloading RetinaFace weights from:\n{self.download_url}")

        try:
            response = requests.get(self.download_url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded_size = 0
            chunk_size = 8192

            with open(self.destination_path, "wb") as output_file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        output_file.write(chunk)
                        downloaded_size += len(chunk)
                        self._print_progress(downloaded_size, total_size)

            print(f"\n[SUCCESS] Model saved to: {self.destination_path}")

        except requests.exceptions.RequestException as error:
            print(f"[ERROR] Failed to download model file: {error}")
            sys.exit(1)

    @staticmethod
    def _print_progress(downloaded: int, total: int) -> None:
        if total > 0:
            percent = (downloaded / total) * 100
            print(f"\r[INFO] Downloaded: {percent:.2f}%", end="")


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_folder_path = os.path.join(project_root, MODEL_DIR_NAME)

    downloader = ModelDownloader(
        download_url=RETINAFACE_WEIGHTS_URL,
        destination_folder=model_folder_path,
        file_name=WEIGHTS_FILE_NAME,
    )

    downloader.download()


if __name__ == "__main__":
    main()