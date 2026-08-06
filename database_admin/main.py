"""
main.py

Single-script terminal admin tool for managing identity_embedding_table:
  1. Add a new person (select image via file picker + enter name)
  2. Read - print the whole table (including facial_data vectors)
  3. Update - re-embed a person's facial_data from a new image
  4. Delete - remove a person by name

Run:
    python main.py
"""

import os
import tkinter as tk
from tkinter import filedialog

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, MetaData, Table, Column, Integer, String, insert, update, delete, select
from sqlalchemy.exc import IntegrityError
from pgvector.sqlalchemy import Vector

from embedding_generation import AdminFaceEmbeddingGenerator


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_CURRENT_DIR, ".env"))

EMBEDDING_DIMENSIONS = 512
SUPPORTED_IMAGE_FILETYPES = [("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]


# --------------------------------------------------------------------- #
# Database setup
# --------------------------------------------------------------------- #

class DatabaseConfig:
    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", "5432"))
        self.database_name = os.getenv("DB_NAME", "IdentityDB")
        self.username = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "")

    def build_connection_url(self):
        return URL.create(
            drivername="postgresql+psycopg2",
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database_name,
        )


_engine = create_engine(DatabaseConfig().build_connection_url(), echo=False)
_metadata = MetaData()

identity_embedding_table = Table(
    "identity_embedding_table",
    _metadata,
    Column("person_unique_id", Integer, primary_key=True),
    Column("person_name", String(150), nullable=False, unique=True),
    Column("facial_data", Vector(EMBEDDING_DIMENSIONS), nullable=False),
)


# --------------------------------------------------------------------- #
# Image file picker
# --------------------------------------------------------------------- #

class ImageFileSelector:
    """Opens a GUI file picker dialog for selecting an image file."""

    @staticmethod
    def select_image_file() -> str:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        selected_path = filedialog.askopenfilename(
            title="Select a person's image",
            filetypes=SUPPORTED_IMAGE_FILETYPES,
        )
        root.destroy()
        return selected_path


# --------------------------------------------------------------------- #
# CRUD services
# --------------------------------------------------------------------- #

class PersonExistenceChecker:
    def __init__(self, engine):
        self.engine = engine

    def exists(self, person_name: str) -> bool:
        statement = select(identity_embedding_table.c.person_unique_id).where(
            identity_embedding_table.c.person_name == person_name
        )
        with self.engine.connect() as connection:
            return connection.execute(statement).fetchone() is not None


class PersonAdminService:
    """Handles Add, Read, Update, Delete operations on identity_embedding_table."""

    def __init__(self):
        self.engine = _engine
        self.existence_checker = PersonExistenceChecker(self.engine)
        self.embedding_generator = AdminFaceEmbeddingGenerator()

    # ------------------------------ ADD ------------------------------ #

    def add_person(self) -> None:
        person_name = input("Enter the person's name: ").strip()
        if not person_name:
            print("[ERROR] Name cannot be empty.")
            return

        if self.existence_checker.exists(person_name):
            print(f"[WARNING] Person '{person_name}' already exists in the database.")
            choice = input("Would you like to update their vectors instead? (yes/no): ").strip().lower()
            if choice in ("yes", "y"):
                self._update_person_embedding(person_name)
            else:
                print("[INFO] Returning to main menu.")
            return

        print("[INFO] Please select an image in the file picker window...")
        image_path = ImageFileSelector.select_image_file()
        if not image_path:
            print("[INFO] No image selected. Cancelled.")
            return

        try:
            embedding = self.embedding_generator.generate_embedding(image_path)
        except (FileNotFoundError, ValueError) as error:
            print(f"[ERROR] {error}")
            return

        try:
            insert_statement = insert(identity_embedding_table).values(
                person_name=person_name,
                facial_data=embedding,
            )
            with self.engine.begin() as connection:
                connection.execute(insert_statement)
            print(f"[SUCCESS] '{person_name}' added successfully.")

        except IntegrityError as error:
            print(f"[ERROR] Failed to add person due to a database constraint: {error.orig}")

    # ------------------------------ READ ------------------------------ #

    def read_all_persons(self) -> None:
        statement = select(identity_embedding_table).order_by(identity_embedding_table.c.person_unique_id)

        with self.engine.connect() as connection:
            rows = connection.execute(statement).fetchall()

        if not rows:
            print("[INFO] No persons found in the database.")
            return

        print(f"\n[INFO] {len(rows)} person(s) found:\n")
        for row in rows:
            print("--------------------------------------------------")
            print(f"person_unique_id : {row.person_unique_id}")
            print(f"person_name      : {row.person_name}")
            print(f"facial_data      : {list(row.facial_data)}")
        print("--------------------------------------------------\n")

    # ----------------------------- UPDATE ----------------------------- #

    def update_person(self) -> None:
        person_name = input("Enter the name of the person to update: ").strip()
        if not person_name:
            print("[ERROR] Name cannot be empty.")
            return

        if not self.existence_checker.exists(person_name):
            print(f"[ERROR] No person found with name '{person_name}'.")
            return

        self._update_person_embedding(person_name)

    def _update_person_embedding(self, person_name: str) -> None:
        print("[INFO] Please select a new image in the file picker window...")
        image_path = ImageFileSelector.select_image_file()
        if not image_path:
            print("[INFO] No image selected. Cancelled.")
            return

        try:
            embedding = self.embedding_generator.generate_embedding(image_path)
        except (FileNotFoundError, ValueError) as error:
            print(f"[ERROR] {error}")
            return

        update_statement = (
            update(identity_embedding_table)
            .where(identity_embedding_table.c.person_name == person_name)
            .values(facial_data=embedding)
        )
        with self.engine.begin() as connection:
            connection.execute(update_statement)

        print(f"[SUCCESS] '{person_name}' vectors updated successfully.")

    # ----------------------------- DELETE ----------------------------- #

    def delete_person(self) -> None:
        person_name = input("Enter the name of the person to delete: ").strip()
        if not person_name:
            print("[ERROR] Name cannot be empty.")
            return

        if not self.existence_checker.exists(person_name):
            print(f"[ERROR] No person found with name '{person_name}'.")
            return

        confirm = input(f"Are you sure you want to delete '{person_name}'? (yes/no): ").strip().lower()
        if confirm not in ("yes", "y"):
            print("[INFO] Delete cancelled.")
            return

        delete_statement = delete(identity_embedding_table).where(
            identity_embedding_table.c.person_name == person_name
        )
        with self.engine.begin() as connection:
            connection.execute(delete_statement)

        print(f"[SUCCESS] '{person_name}' deleted successfully.")


# --------------------------------------------------------------------- #
# Terminal menu
# --------------------------------------------------------------------- #

class AdminMenu:
    """Interactive terminal menu for the four CRUD operations."""

    def __init__(self):
        self.service = PersonAdminService()

    def run(self) -> None:
        while True:
            self._print_menu()
            choice = input("Select an option (1-5): ").strip()

            if choice == "1":
                self.service.add_person()
            elif choice == "2":
                self.service.read_all_persons()
            elif choice == "3":
                self.service.update_person()
            elif choice == "4":
                self.service.delete_person()
            elif choice == "5":
                print("[INFO] Exiting. Goodbye!")
                break
            else:
                print("[ERROR] Invalid choice. Please select 1-5.")

    @staticmethod
    def _print_menu() -> None:
        print("\n==================== DATABASE ADMIN ====================")
        print("1. Add new person")
        print("2. Read all persons")
        print("3. Update person's facial data")
        print("4. Delete person")
        print("5. Exit")
        print("==========================================================")


def main():
    menu = AdminMenu()
    menu.run()


if __name__ == "__main__":
    main()