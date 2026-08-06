"""
test_logger.py

Standalone logger for authentication testing. Unlike log.py (which
only logs successful authentications), this logs BOTH successful and
unsuccessful attempts to test_log.json, so a full test session can
later be evaluated for accuracy using metric.py.

Kept standalone for now — wire TestLogger into your live testing
flow manually whenever you're ready to run an evaluation session.
"""

import os
import json
import uuid
from datetime import datetime, timezone


AUTHENTICATION_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_LOGS_DIR = os.path.join(AUTHENTICATION_DIR, "logs")
TEST_LOG_FILE_PATH = os.path.join(TEST_LOGS_DIR, "test_log.json")

STATUS_SUCCESSFUL = "Successful"
STATUS_UNSUCCESSFUL = "Unsuccessful"


class TestLogger:
    """
    Appends every authentication attempt (Successful or Unsuccessful)
    to test_log.json, each with a unique attempt_id so individual
    entries can later be referenced/labeled by metric.py.
    """

    def __init__(self, log_file_path: str = TEST_LOG_FILE_PATH):
        self.log_file_path = log_file_path
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

    def log_successful_attempt(self, person_id, person_name: str, distance: float = None) -> dict:
        return self._log_attempt(
            status=STATUS_SUCCESSFUL,
            person_id=person_id,
            person_name=person_name,
            distance=distance,
        )

    def log_unsuccessful_attempt(self, distance: float = None) -> dict:
        return self._log_attempt(
            status=STATUS_UNSUCCESSFUL,
            person_id=None,
            person_name=None,
            distance=distance,
        )

    def _log_attempt(self, status: str, person_id, person_name, distance) -> dict:
        log_entry = {
            "attempt_id": str(uuid.uuid4()),
            "status": status,
            "person_id": person_id,
            "person_name": person_name,
            "distance": distance,
            "time_detected": datetime.now(timezone.utc).isoformat(),
        }

        existing_logs = self._load_existing_logs()
        existing_logs.append(log_entry)

        with open(self.log_file_path, "w") as log_file:
            json.dump(existing_logs, log_file, indent=4)

        return log_entry

    def _load_existing_logs(self) -> list:
        if not os.path.isfile(self.log_file_path):
            return []
        try:
            with open(self.log_file_path, "r") as log_file:
                data = json.load(log_file)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []