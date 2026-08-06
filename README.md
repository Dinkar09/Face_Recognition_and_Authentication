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
   face.
3. **Authentication** — When a face is detected, the system checks it against everyone
   who's been registered.
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
python facenet512_model_download.py
python retinaface_model_download.py
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

A camera window will open. Show your face to the camera, and the system will tell you
whether it recognizes you.

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
| `vector_store` | Saved face data generated during registration |
