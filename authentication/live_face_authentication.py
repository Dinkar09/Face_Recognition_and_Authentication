"""
live_face_authentication.py

Microservice 2 - Face Detection and Tracking (live video), with
manual authentication triggering and multithreaded authentication
so the camera window never freezes while waiting on the embedding
generation + API call.

Continuously tracks a face using YuNet (fast, runs every frame).
Once boxed, the screen shows "Press Space to detect". On Space press,
the authentication flow (RetinaFace + FaceNet512 embedding, then API
call) runs in a BACKGROUND THREAD, while the main loop keeps reading
frames and calling cv2.waitKey() — keeping the window responsive.
Results are handed back to the main thread via a thread-safe queue.

Press 'q' to quit at any time.
"""

import os
import sys
import time
import threading
import queue

import cv2
import requests
from dotenv import load_dotenv

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from live_embedding_pipeline import LiveFaceEmbeddingGenerator  # noqa: E402
from log import AuthenticationLogger  # noqa: E402
from test_logger import TestLogger  # noqa: E402

load_dotenv(dotenv_path=os.path.join(_CURRENT_DIR, ".env"))

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
AUTHENTICATE_URL = f"http://{API_HOST}:{API_PORT}/authenticate"

YUNET_MODEL_PATH = os.path.join(_MODELS_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_SCORE_THRESHOLD = 0.65
YUNET_NMS_THRESHOLD = 0.30
YUNET_TOP_K = 5000

RESULT_DISPLAY_SECONDS = 1
WINDOW_NAME = "Face Recognition and Authentication"
BOX_COLOR_GREEN = (0, 210, 80)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class YuNetFaceDetector:
    """Detects the single most confident face in a frame using YuNet."""

    def __init__(self, model_path: str = YUNET_MODEL_PATH):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"YuNet model not found at: {model_path}\n"
                "Run: python models/yunet_model_download.py"
            )

        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            config="",
            input_size=(320, 320),
            score_threshold=YUNET_SCORE_THRESHOLD,
            nms_threshold=YUNET_NMS_THRESHOLD,
            top_k=YUNET_TOP_K,
        )

    def detect_best_face_box(self, frame):
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0:
            return None

        best_face = max(faces, key=lambda face_row: face_row[-1])
        x, y, w, h = best_face[:4].astype(int)
        return (x, y, x + w, y + h)


class AuthenticationApiClient:
    def __init__(self, url: str = AUTHENTICATE_URL):
        self.url = url

    def authenticate(self, embedding: list) -> dict:
        response = requests.post(self.url, json={"embedding": embedding}, timeout=10)
        response.raise_for_status()
        return response.json()


class ScreenOverlay:
    @staticmethod
    def draw_tracking_box(frame, box):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR_GREEN, 2)
        cv2.putText(frame, "Press Space to detect", (x1, max(y1 - 12, 20)),
                    FONT, 0.7, BOX_COLOR_GREEN, 2, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_processing_screen(frame):
        overlay_frame = frame.copy()
        cv2.putText(overlay_frame, "Processing...", (40, 60), FONT, 1.2, (255, 255, 255), 3)
        return overlay_frame

    @staticmethod
    def draw_authenticated_screen(frame, person_name: str):
        overlay_frame = frame.copy()
        cv2.putText(overlay_frame, f"Person: {person_name} is Authenticated", (30, 60),
                    FONT, 0.9, (0, 200, 0), 3)
        return overlay_frame

    @staticmethod
    def draw_not_authenticated_screen(frame):
        overlay_frame = frame.copy()
        cv2.putText(overlay_frame, "Person: Unknown is not Authenticated", (30, 60),
                    FONT, 0.9, (0, 0, 255), 3)
        cv2.putText(overlay_frame, "Try Again", (30, 100), FONT, 0.9, (0, 0, 255), 3)
        return overlay_frame


class LiveAuthenticationApp:
    """
    Tracks a face continuously with a green box (YuNet, fast, runs
    every frame on the main thread). Authentication (RetinaFace +
    FaceNet512 + API call) runs in a background thread on Space press,
    so the camera feed never freezes while waiting on it.
    """

    def __init__(self):
        self.face_detector = YuNetFaceDetector()
        self.embedding_generator = LiveFaceEmbeddingGenerator()
        self.api_client = AuthenticationApiClient()
        self.logger = AuthenticationLogger()
        self.test_logger = TestLogger()
        self.video_capture = cv2.VideoCapture(0)
        self.current_face_box = None

        # Threading / synchronization
        self.result_queue = queue.Queue()   # thread-safe hand-off, no manual lock needed
        self.is_processing = False

    def run(self) -> None:
        if not self.video_capture.isOpened():
            raise RuntimeError("Could not open webcam.")

        print("[INFO] Live face authentication started.")
        print("[INFO] Press SPACE to authenticate the boxed face. Press 'q' to quit.")

        result_display_until = 0.0
        last_result_frame = None

        try:
            while True:
                success, frame = self.video_capture.read()
                if not success:
                    print("[WARNING] Failed to read frame from camera.")
                    continue

                # Check if a background authentication attempt just finished
                try:
                    result_payload = self.result_queue.get_nowait()
                    self.is_processing = False
                    last_result_frame = result_payload["frame_for_display"]
                    result_display_until = time.time() + RESULT_DISPLAY_SECONDS
                except queue.Empty:
                    pass

                if last_result_frame is not None and time.time() < result_display_until:
                    cv2.imshow(WINDOW_NAME, last_result_frame)
                elif self.is_processing:
                    cv2.imshow(WINDOW_NAME, ScreenOverlay.draw_processing_screen(frame))
                else:
                    self.current_face_box = self.face_detector.detect_best_face_box(frame)
                    display_frame = frame.copy()
                    if self.current_face_box is not None:
                        display_frame = ScreenOverlay.draw_tracking_box(display_frame, self.current_face_box)
                    cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" ") and not self.is_processing:
                    if self.current_face_box is not None:
                        self._trigger_authentication_async(frame)
                    else:
                        print("[INFO] Space pressed, but no face is currently boxed. Ignored.")

        finally:
            self.video_capture.release()
            cv2.destroyAllWindows()

    def _trigger_authentication_async(self, frame) -> None:
        """Starts authentication in a background thread; UI keeps running."""
        self.is_processing = True
        worker_thread = threading.Thread(target=self._authenticate_in_background, args=(frame,), daemon=True)
        worker_thread.start()

    def _authenticate_in_background(self, frame) -> None:
        """Runs on a background thread. Never touches cv2 window calls directly."""
        try:
            embedding = self.embedding_generator.generate_embedding(frame)
            result = self.api_client.authenticate(embedding)

            if result.get("status") == "Successful":
                print(f"[SUCCESS] {result.get('message')}")
                self.logger.log_authentication(result.get("person_unique_id"), result.get("person_name"))
                self.test_logger.log_successful_attempt(
                    person_id=result.get("person_unique_id"),
                    person_name=result.get("person_name"),
                    distance=result.get("distance"),
                )
                result_frame = ScreenOverlay.draw_authenticated_screen(frame, result.get("person_name"))
            else:
                print(f"[INFO] {result.get('message')}")
                self.test_logger.log_unsuccessful_attempt(distance=result.get("distance"))
                result_frame = ScreenOverlay.draw_not_authenticated_screen(frame)

        except (ValueError, requests.RequestException) as error:
            print(f"[ERROR] {error}")
            self.test_logger.log_unsuccessful_attempt(distance=None)
            result_frame = ScreenOverlay.draw_not_authenticated_screen(frame)

        # queue.Queue.put() is thread-safe — safe hand-off back to the main thread
        self.result_queue.put({"frame_for_display": result_frame})


def main():
    app = LiveAuthenticationApp()
    app.run()


if __name__ == "__main__":
    main()