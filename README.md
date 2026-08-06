# Face Recognition and Authentication

## What This Project Does

This project recognizes people's faces in real time using a webcam.

You first **register** people by adding their photos. The system learns what each
person looks like and stores that information securely. After that, you can point a
webcam at someone, and the system will tell you whether it recognizes them — showing
their name on screen if they're a match, or a "not recognized" message if they're not.

Think of it like a smart face-based check-in system: register once, then get instantly
recognized afterward.

---

## How It Works (Simple Overview)

1. **Registration** — You place a photo of each person in a folder. The system looks at
   each photo, learns the person's face, and saves that information to a database.
2. **Live Camera Detection** — A webcam window opens and continuously watches for a
   face, drawing a green box around it once found.
3. **Authentication** — Pressing the **Space bar** checks the boxed face against
   everyone who's been registered.
   - ✅ If it recognizes the person, it shows their name in green:
     *"Person: [Name] is Authenticated."*
   - ❌ If it doesn't recognize them, it shows a message in red asking them to try
     again.
4. The camera then automatically resumes watching for the next person.

---

## Before You Start

Make sure you have:
- **Python** installed on your computer.
- **Docker Desktop** installed and running (this runs the database in the background).
- A working **webcam**.

---

## How to Run the Project

### Step 1 — Set up your Python environment

Open a terminal in the project folder and run:

```bash
python -m venv my_env
my_env\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Download the face recognition models

```bash
cd models
python download_all_models.py
cd ..
```

### Step 3 — Start the database

```bash
docker compose up -d
```

*(The first time only: open pgAdmin and create the database and table — ask if you
need the setup steps again.)*

### Step 4 — Register people

1. Add a photo for each person into the `images` folder. Name each photo file after
   the person (e.g. `John.png`).
2. Run:

```bash
cd embedding
python all_embedding_generation.py
```

This teaches the system what each person looks like.

### Step 5 — Start the recognition service

Open a terminal and run:

```bash
cd authentication
python authentication_api.py
```

Leave this terminal running in the background.

### Step 6 — Start the live camera

Open a **second** terminal and run:

```bash
cd authentication
python live_face_authentication.py
```

A camera window will open and draw a green box around any face it sees. Press
**Space** to check that face against the registered people.

To stop, press **`q`** while the camera window is active.

---

## Checking How Accurate the System Is

If you'd like to see how well the system is performing, you can run a simple check
that asks you to confirm whether it identified people correctly, and then gives you an
accuracy percentage:

```bash
cd authentication
python metric.py
```

---

## Project Folders (Quick Reference)

| Folder | What's Inside |
|---|---|
| `images` | Photos of people to register |
| `models` | The AI models used to recognize faces |
| `embedding` | Tools that teach the system new faces |
| `authentication` | The live camera and recognition service |
| `database_admin` | A tool to add, view, update, or remove people from the database |
| `vector_store` | Saved face data generated during registration |

---

## Version History

### Version 1 — Original Working System

The first complete version of the project. Everything ran one step at a time:

- Registering a person's photo, checking the camera for a face, and asking the
  recognition service for a match all happened one after another, waiting for each
  step to finish before starting the next.
- Registering a large batch of people took a while, since each person was processed
  one by one.
- The camera window would briefly freeze while waiting for a recognition result.
- Downloading the three AI models happened one at a time.

This version worked reliably, but had some natural slowness built in from doing
everything sequentially.

### Version 2 — Performance Improvements (Current)

The same features as Version 1, but reworked so independent tasks can happen **at the
same time** instead of waiting in line:

- **Faster bulk registration** — When registering many people at once, several
  people's photos are now processed simultaneously instead of one at a time.
- **Smoother live camera** — The camera window no longer freezes while checking a
  face. It stays smooth and responsive even while a recognition result is being
  fetched in the background.
- **Faster model setup** — All three AI models now download at the same time instead
  of one after another.
- **A more efficient recognition service** — The service that matches faces against
  the database was upgraded to handle multiple requests more efficiently.

**Files changed for Version 2:**
- `authentication/authentication_api.py`
- `authentication/live_face_authentication.py`
- `embedding/embedding_tracing.py`
- `embedding/all_embedding_generation.py`
- `models/download_all_models.py` *(new file)*

No steps to run the project changed between versions — everything in the
[How to Run](#how-to-run-the-project) section above applies to the current
(Version 2) codebase.