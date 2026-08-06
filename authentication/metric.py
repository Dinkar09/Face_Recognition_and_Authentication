"""
metric.py

Evaluates authentication accuracy using test_log.json (produced by
test.py). For every "Successful" attempt, asks the user in the
terminal whether the identified person was actually correct
(TRUE/FALSE) — since the system can report "Successful" while still
having matched the WRONG registered person.

Human answers are saved to metric_results.json (keyed by attempt_id),
so re-running this script resumes instead of re-asking already
labeled entries.

Run separately, whenever you want to score a completed test session:
    python metric.py
"""

import os
import json

from test_logger import TEST_LOG_FILE_PATH, STATUS_SUCCESSFUL, STATUS_UNSUCCESSFUL


AUTHENTICATION_DIR = os.path.dirname(os.path.abspath(__file__))
METRIC_RESULTS_FILE_PATH = os.path.join(AUTHENTICATION_DIR, "logs", "metric_results.json")

VALID_TRUE_INPUTS = {"true", "t", "yes", "y"}
VALID_FALSE_INPUTS = {"false", "f", "no", "n"}


class MetricResultsStore:
    """Persists human TRUE/FALSE labels per attempt_id across runs."""

    def __init__(self, results_file_path: str = METRIC_RESULTS_FILE_PATH):
        self.results_file_path = results_file_path
        os.makedirs(os.path.dirname(self.results_file_path), exist_ok=True)
        self.results = self._load_existing_results()

    def has_label(self, attempt_id: str) -> bool:
        return attempt_id in self.results

    def get_label(self, attempt_id: str) -> bool:
        return self.results[attempt_id]

    def save_label(self, attempt_id: str, is_correct: bool) -> None:
        self.results[attempt_id] = is_correct
        self._persist()

    def _load_existing_results(self) -> dict:
        if not os.path.isfile(self.results_file_path):
            return {}
        try:
            with open(self.results_file_path, "r") as results_file:
                data = json.load(results_file)
                return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _persist(self) -> None:
        with open(self.results_file_path, "w") as results_file:
            json.dump(self.results, results_file, indent=4)


class HumanLabelCollector:
    """Prompts the user in the terminal to confirm each Successful identification."""

    @staticmethod
    def ask_is_identification_correct(entry: dict) -> bool:
        print("\n--------------------------------------------------")
        print(f"Attempt ID     : {entry['attempt_id']}")
        print(f"Time Detected  : {entry['time_detected']}")
        print(f"Identified As  : {entry['person_name']} (person_id={entry['person_id']})")
        print(f"Match Distance : {entry.get('distance')}")

        while True:
            raw_input_value = input("Was this identification correct? (TRUE/FALSE): ").strip().lower()
            if raw_input_value in VALID_TRUE_INPUTS:
                return True
            if raw_input_value in VALID_FALSE_INPUTS:
                return False
            print("Please enter TRUE or FALSE (or T/F, Y/N).")


class AuthenticationAccuracyEvaluator:
    """
    Loads test_log.json, collects/reuses human labels for every
    Successful attempt, and computes accuracy metrics.
    """

    def __init__(self):
        self.results_store = MetricResultsStore()
        self.label_collector = HumanLabelCollector()

    def run(self) -> None:
        all_entries = self._load_test_log()

        if not all_entries:
            print(f"[WARNING] No entries found in {TEST_LOG_FILE_PATH}. Run test.py logging first.")
            return

        successful_entries = [entry for entry in all_entries if entry["status"] == STATUS_SUCCESSFUL]
        unsuccessful_entries = [entry for entry in all_entries if entry["status"] == STATUS_UNSUCCESSFUL]

        self._collect_missing_labels(successful_entries)
        self._print_report(all_entries, successful_entries, unsuccessful_entries)

    def _load_test_log(self) -> list:
        if not os.path.isfile(TEST_LOG_FILE_PATH):
            return []
        with open(TEST_LOG_FILE_PATH, "r") as test_log_file:
            data = json.load(test_log_file)
            return data if isinstance(data, list) else []

    def _collect_missing_labels(self, successful_entries: list) -> None:
        unlabeled_entries = [
            entry for entry in successful_entries
            if not self.results_store.has_label(entry["attempt_id"])
        ]

        if not unlabeled_entries:
            print("[INFO] All Successful attempts already labeled. Skipping straight to report.")
            return

        print(f"[INFO] {len(unlabeled_entries)} Successful attempt(s) need labeling.\n")

        for entry in unlabeled_entries:
            is_correct = self.label_collector.ask_is_identification_correct(entry)
            self.results_store.save_label(entry["attempt_id"], is_correct)

    def _print_report(self, all_entries: list, successful_entries: list, unsuccessful_entries: list) -> None:
        total_attempts = len(all_entries)
        successful_count = len(successful_entries)
        unsuccessful_count = len(unsuccessful_entries)

        correct_identifications = sum(
            1 for entry in successful_entries
            if self.results_store.get_label(entry["attempt_id"]) is True
        )
        wrong_identifications = successful_count - correct_identifications

        successful_rate = (successful_count / total_attempts * 100) if total_attempts else 0.0
        identification_accuracy = (
            correct_identifications / successful_count * 100
        ) if successful_count else 0.0
        misidentification_rate = (
            wrong_identifications / successful_count * 100
        ) if successful_count else 0.0
        overall_accuracy = (
            correct_identifications / total_attempts * 100
        ) if total_attempts else 0.0

        print("\n==================== ACCURACY REPORT ====================")
        print(f"Total Attempts                       : {total_attempts}")
        print(f"Successful Detections                 : {successful_count}")
        print(f"Unsuccessful Detections               : {unsuccessful_count}")
        print(f"Successful Detection Rate             : {successful_rate:.2f}%  "
              f"(Successful / (Successful + Unsuccessful))")
        print("-----------------------------------------------------------")
        print(f"Human-Verified Correct IDs (TRUE)     : {correct_identifications}")
        print(f"Human-Verified Wrong IDs (FALSE)      : {wrong_identifications}")
        print(f"Identification Accuracy               : {identification_accuracy:.2f}%  "
              f"(Correct / Successful) -- Precision / Rank-1 Accuracy")
        print(f"Misidentification Rate                : {misidentification_rate:.2f}%  "
              f"(Wrong / Successful)")
        print("-----------------------------------------------------------")
        print(f"Overall System Accuracy               : {overall_accuracy:.2f}%  "
              f"(Correctly Identified / Total Attempts)")
        print("=============================================================\n")


def main():
    evaluator = AuthenticationAccuracyEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()