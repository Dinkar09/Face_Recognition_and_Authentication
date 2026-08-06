"""
download_all_models.py

Downloads all three pretrained model files (FaceNet512, RetinaFace,
YuNet) in parallel using a thread pool, since they're independent
I/O-bound downloads with no shared state.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import facenet512_model_download
import retinaface_model_download
import yunet_model_download


def run_all_downloads() -> None:
    download_tasks = {
        "FaceNet512": facenet512_model_download.main,
        "RetinaFace": retinaface_model_download.main,
        "YuNet": yunet_model_download.main,
    }

    print(f"[INFO] Starting {len(download_tasks)} downloads in parallel...")

    with ThreadPoolExecutor(max_workers=len(download_tasks)) as executor:
        futures = {executor.submit(task): name for name, task in download_tasks.items()}

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                future.result()
                print(f"[SUCCESS] {model_name} download finished.")
            except Exception as error:
                print(f"[ERROR] {model_name} download failed: {error}")

    print("[INFO] All downloads complete.")


if __name__ == "__main__":
    run_all_downloads()