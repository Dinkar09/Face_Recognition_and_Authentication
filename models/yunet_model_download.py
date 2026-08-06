"""
yunet_model_download.py

Downloads the pretrained YuNet face detection model (ONNX format,
from OpenCV Zoo) and stores it inside the /models folder of the
current project.

File type downloaded: .onnx
Used by: authentication/live_face_tracking.py (lightweight live
face detection for the continuous green bounding-box preview).
"""

import os
import sys
import requests

YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

MODEL_FILE_NAME = "face_detection_yunet_2023mar.onnx"


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
        print(f"[INFO] Downloading YuNet model from:\n{self.download_url}")

        try:
            response = requests.get(self.download_url, stream=True, timeout=60)
            response.raise_for_status()

            with open(self.destination_path, "wb") as output_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        output_file.write(chunk)

            print(f"[SUCCESS] Model saved to: {self.destination_path}")

        except requests.exceptions.RequestException as error:
            print(f"[ERROR] Failed to download model file: {error}")
            sys.exit(1)


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    downloader = ModelDownloader(
        download_url=YUNET_MODEL_URL,
        destination_folder=project_root,
        file_name=MODEL_FILE_NAME,
    )
    downloader.download()


if __name__ == "__main__":
    main()