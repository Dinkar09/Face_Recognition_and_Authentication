"""
live_face_authentication.py

Microservice 2 - Face Detection and Tracking (live video), merged
with manual authentication triggering.

Continuously tracks a face in the live webcam feed using YuNet
(OpenCV's lightweight ONNX face detector) — fast enough to run every
frame, drawing a green bounding box with no wasted API calls.

Once a face is boxed, the screen shows "Press Space to detect". Only
when Space is pressed (with a face currently boxed) does the full
authentication flow run:
  1. The frame freezes, showing "Processing...".
  2. A 512D embedding is generated using RetinaFace + FaceNet512
     (live_embedding_pipeline.py) — keeps compatibility with the
     existing database.
  3. The embedding is sent to authentication_api.py's /authenticate
     endpoint.
  4. Result is shown in green ("Person: X is Authenticated") or red
     ("Person: X is not Authenticated / Try Again") for a short delay.
  5. Successful authentications are logged via log.py; both outcomes
     are logged via test_logger.py.
  6. Tracking then resumes automatically.

Press 'q' to quit at any time.
"""

import os
import sys
import time

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

RESULT_DISPLAY_SECONDS = 3
WINDOW_NAME = "Face Recognition and Authentication"
BOX_COLOR_GREEN = (0, 210, 80)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------- #
# Lightweight face detector (YuNet) — used every frame for the live box
# --------------------------------------------------------------------- #

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
        """
        Returns (x1, y1, x2, y2) of the highest-confidence face, or
        None if no face is detected.
        """
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0:
            return None

        best_face = max(faces, key=lambda face_row: face_row[-1])
        x, y, w, h = best_face[:4].astype(int)
        return (x, y, x + w, y + h)


# --------------------------------------------------------------------- #
# Authentication API client
# --------------------------------------------------------------------- #

class AuthenticationApiClient:
    def __init__(self, url: str = AUTHENTICATE_URL):
        self.url = url

    def authenticate(self, embedding: list) -> dict:
        response = requests.post(self.url, json={"embedding": embedding}, timeout=10)
        response.raise_for_status()
        return response.json()


# --------------------------------------------------------------------- #
# Overlay helpers
# --------------------------------------------------------------------- #

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


# --------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------- #

class LiveAuthenticationApp:
    """
    Tracks a face continuously with a green box (YuNet, fast). Once
    boxed, prompts "Press Space to detect". Authentication (RetinaFace
    + FaceNet512 + API call) only runs on Space press.
    """

    def __init__(self):
        self.face_detector = YuNetFaceDetector()
        self.embedding_generator = LiveFaceEmbeddingGenerator()
        self.api_client = AuthenticationApiClient()
        self.logger = AuthenticationLogger()
        self.test_logger = TestLogger()
        self.video_capture = cv2.VideoCapture(0)
        self.current_face_box = None

    def run(self) -> None:
        if not self.video_capture.isOpened():
            raise RuntimeError("Could not open webcam.")

        print("[INFO] Live face authentication started.")
        print("[INFO] Press SPACE to authenticate the boxed face. Press 'q' to quit.")

        try:
            while True:
                success, frame = self.video_capture.read()
                if not success:
                    print("[WARNING] Failed to read frame from camera.")
                    continue

                self.current_face_box = self.face_detector.detect_best_face_box(frame)

                display_frame = frame.copy()
                if self.current_face_box is not None:
                    display_frame = ScreenOverlay.draw_tracking_box(display_frame, self.current_face_box)

                cv2.imshow(WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    if self.current_face_box is not None:
                        self._authenticate_current_frame(frame)
                    else:
                        print("[INFO] Space pressed, but no face is currently boxed. Ignored.")

        finally:
            self.video_capture.release()
            cv2.destroyAllWindows()

    def _authenticate_current_frame(self, frame) -> None:
        processing_frame = ScreenOverlay.draw_processing_screen(frame)
        cv2.imshow(WINDOW_NAME, processing_frame)
        cv2.waitKey(1)

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

        self._show_result_for_delay(result_frame)

    def _show_result_for_delay(self, result_frame, delay_seconds: int = RESULT_DISPLAY_SECONDS) -> None:
        end_time = time.time() + delay_seconds
        while time.time() < end_time:
            cv2.imshow(WINDOW_NAME, result_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main():
    app = LiveAuthenticationApp()
    app.run()


if __name__ == "__main__":
    main()