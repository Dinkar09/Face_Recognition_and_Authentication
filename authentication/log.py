"""
log.py

Logs each successful live authentication event to a JSON file, so
there's a simple audit trail of who was detected and when.

Each entry: {person_id, person_name, time_detected}
Stored in authentication/logs/authentication_log.json as a JSON
array, appended to on every logged event.
"""

import os
import json
from datetime import datetime, timezone


AUTHENTICATION_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(AUTHENTICATION_DIR, "logs")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "authentication_log.json")


class AuthenticationLogger:
    """
    Appends authentication events (person_id, person_name,
    time_detected) to a JSON log file.
    """

    def __init__(self, log_file_path: str = LOG_FILE_PATH):
        self.log_file_path = log_file_path
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

    def log_authentication(self, person_id, person_name: str) -> dict:
        """
        Records a single authentication event.

        Returns:
            dict: The log entry that was written.
        """
        log_entry = {
            "person_id": person_id,
            "person_name": person_name,
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