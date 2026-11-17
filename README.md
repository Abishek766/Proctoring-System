# Video Proctoring Engine

A real-time AI-powered exam proctoring backend that monitors video and audio streams for academic integrity violations. Built with FastAPI and WebSockets, it runs multiple ML pipelines in parallel to detect suspicious behaviour during online assessments.

---

## Features

| Category | What it detects |
|---|---|
| **Head Pose** | Candidate turning left, right, up, or down beyond configurable angle thresholds |
| **Eye Gaze** | Candidate looking away from the screen (iris tracking via MediaPipe FaceMesh) |
| **Person Count** | No candidate visible, or more than one person in frame |
| **Prohibited Objects** | Cell phones, laptops, tablets, books, earphones, keyboards (YOLOv8) |
| **External Hands** | A second pair of hands entering the frame |
| **Audio Anomalies** | External speech or background device sounds (YAMNet), with lip-sync suppression to avoid false positives from the candidate speaking |
| **Mouth Detection** | Candidate speaking — Mouth Aspect Ratio (MAR) thresholding |
| **Face Identity** | Periodic and violation-triggered identity verification against a stored 512-D ArcFace embedding |
| **Background Validation** | Detects if the candidate has changed exam location using DINOv2 scene embeddings |

---

## Architecture

```
Client (browser / mobile)
        │  WebSocket  (ws://.../ws/proctor/{session_id})
        │  30 FPS JPEG frames + optional float32 PCM audio
        ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI  (proctoring_engine/app.py)                │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ auth_routes │  │ verification │  │  ws route │  │
│  │  /register  │  │   /verify    │  │  /ws/     │  │
│  │  /login     │  │              │  │  proctor  │  │
│  └─────────────┘  └──────────────┘  └─────┬─────┘  │
│                                           │        │
│                                    SessionManager  │
│                                           │        │
│                               ┌───────────▼──────┐ │
│                               │  InferenceEngine │ │
│                               │                  │ │
│                               │  ┌─────────────┐ │ │
│                               │  │ FacePipeline│ │ │
│                               │  │ (MediaPipe) │ │ │
│                               │  ├─────────────┤ │ │
│                               │  │ HandPipeline│ │ │
│                               │  ├─────────────┤ │ │
│                               │  │ YoloPipeline│ │ │
│                               │  │  (YOLOv8s)  │ │ │
│                               │  ├─────────────┤ │ │
│                               │  │AudioPipeline│ │ │
│                               │  │  (YAMNet)   │ │ │
│                               │  └─────────────┘ │ │
│                               └──────────────────┘ │
│                                                     │
│  PostgreSQL + pgvector  (face & background vectors) │
└─────────────────────────────────────────────────────┘
```

Each WebSocket connection owns one `InferenceEngine` instance. The engine accumulates 6 frames (~1 second at 30 FPS) into a window, then sends a `WindowResponse` with all active violations and their fade-out severity scores.

---

## Tech Stack

| Layer | Library / Model |
|---|---|
| API framework | FastAPI 0.115, Uvicorn 0.32, WebSockets 13 |
| Face landmarks (pose + gaze) | MediaPipe FaceMesh 0.10 |
| Hand detection | MediaPipe Hands |
| Person segmentation | MediaPipe Selfie Segmentation |
| Object detection | YOLOv8s (Ultralytics 8.3) |
| Audio classification | YAMNet via TensorFlow Hub 0.16 / TF 2.17 |
| Face recognition | InsightFace 0.7 (ArcFace, `buffalo_l`, 512-D embeddings) |
| Background embeddings | DINOv2 ViT-B/14 (PyTorch, 768-D embeddings) |
| Vector similarity | pgvector (PostgreSQL extension) |
| Database ORM | SQLAlchemy 2 async + asyncpg |
| Auth | bcrypt passwords + JWT (python-jose) |
| Settings | pydantic-settings + `.env` |

---

## Project Structure

```
src/
├── requirements.txt
└── proctoring_engine/
    ├── main.py                   # Camera benchmark entry point
    ├── api.py                    # API server entry point
    ├── app.py                    # FastAPI app factory + lifespan
    ├── settings.py               # All configuration (env vars)
    ├── core/
    │   ├── face_pipeline.py      # Head pose + eye gaze (MediaPipe FaceMesh)
    │   ├── pose_pipeline.py      # Hand detection (MediaPipe Hands)
    │   ├── yolo_pipeline.py      # Prohibited object detection (YOLOv8)
    │   ├── audio_pipeline.py     # Audio classification (YAMNet)
    │   ├── background_pipeline.py# Background change detection (DINOv2)
    │   ├── face_detection.py     # Face detection utilities
    │   ├── face_recognition.py   # ArcFace embedding generation
    │   ├── image_processing.py   # Image utilities
    │   ├── image_validation.py   # Quality checks (brightness, sharpness)
    │   └── model_loader.py       # Centralised model pre-loading
    ├── services/
    │   ├── inference_engine.py   # Stateless per-frame inference orchestrator
    │   ├── session_manager.py    # WebSocket session lifecycle
    │   ├── proctoring_service.py # Camera-mode proctoring loop
    │   ├── face_service.py       # Face embedding service
    │   ├── face_verification_service.py # Verification logic
    │   ├── auth_service.py       # Register / login
    │   ├── background_service.py # Background validation
    │   └── dependencies.py       # FastAPI dependency injection
    ├── routes/
    │   ├── proctoring_ws.py      # WebSocket endpoint
    │   ├── auth_routes.py        # /register, /login
    │   ├── verification_routes.py# /verify
    │   └── default.py            # Health check
    ├── models/
    │   ├── violations.py         # Pydantic violation event models
    │   ├── api_models.py         # Wire-format request/response models
    │   ├── schemas.py            # Auth/verification schemas
    │   ├── database.py           # SQLAlchemy ORM models
    │   └── metrics.py            # Benchmark metrics
    ├── database/
    │   └── connection.py         # Async SQLAlchemy engine + session
    ├── repositories/
    │   └── user_repository.py    # User CRUD
    └── utils/
        ├── constants.py
        ├── jwt_handler.py
        ├── logger.py
        └── exceptions/           # Typed exceptions + error codes
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL with the `pgvector` extension enabled
- A YOLOv8s model file (`yolov8s.pt`) placed at `src/yolov8s.pt`
- For GPU inference: CUDA-compatible PyTorch (see requirements)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/video-proctoring-engine.git
cd video-proctoring-engine

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install CPU PyTorch first (or GPU variant — see requirements.txt comments)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
pip install -r src/requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/video_proctoring

# JWT
SECRET_KEY=change-me-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Optional overrides (defaults shown)
YOLO_MODEL=src/yolov8s.pt
YOLO_CONFIDENCE_THRESHOLD=0.35
HEAD_POSE_YAW_THRESHOLD=20.0
EYE_GAZE_LEFT_THRESHOLD=0.35
FACE_SIMILARITY_THRESHOLD=0.70
BG_SIMILARITY_THRESHOLD=0.85
VERIFICATION_INTERVAL_SECONDS=60
```

### Database Setup

```sql
-- In psql
CREATE DATABASE video_proctoring;
\c video_proctoring
CREATE EXTENSION IF NOT EXISTS vector;
```

Tables are created automatically on first startup via SQLAlchemy's `metadata.create_all`.

### Run the API Server

```bash
python -m proctoring_engine.api
```

The server starts on `http://0.0.0.0:8001`. On first startup all seven models are pre-loaded and warmed up before any connection is accepted. This takes 30–60 seconds depending on hardware and whether models are already cached.

### Run the Camera Benchmark (local webcam mode)

```bash
python -m proctoring_engine.main
```

Opens a local webcam feed with an overlay window. Press `q` to stop.

---

## API Reference

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/register` | Register a new candidate with username, email, password, and profile photo |
| `POST` | `/api/v1/login` | Authenticate and receive a JWT access token |

**Register** accepts `multipart/form-data`:
- `username` (string)
- `email` (string)
- `password` (string)
- `profile_photo` (file) — JPEG/PNG, used to generate and store a 512-D ArcFace embedding

**Login** accepts JSON:
```json
{ "email": "user@example.com", "password": "secret" }
```

### Face Verification

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/verify` | Verify a face image against a stored embedding |

Accepts `multipart/form-data`:
- `user_id` (string) — UUID from registration
- `image` (file) — JPEG/PNG face image
- `source_type` (string) — `"verify"` (full quality checks) or `"violation_verify"` (skips quality checks, used internally during sessions)

Returns:
```json
{
  "status_code": 200,
  "verified": true,
  "similarity_score": 0.85,
  "message": "Candidate verified successfully"
}
```

### Health Check

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` or `/health` | Returns server status |

### Proctoring WebSocket

```
ws://<host>:8001/ws/proctor/{session_id}?user_id=<uuid>
```

**Connection flow:**

1. Client connects. Server immediately sends a `SessionStartResponse`.
2. Client streams frames at 30 FPS as JSON text messages:

```json
{
  "session_id": "candidate-001",
  "frame_id": 1,
  "timestamp_ms": 1700000000000,
  "frame_b64": "<base64 JPEG>",
  "audio_b64": "<base64 float32 PCM, optional>"
}
```

Audio can also be sent as raw binary WebSocket messages (float32 PCM bytes, 16 kHz mono, no header).

3. After every 6 frames (~1 second), the server sends a `WindowResponse`:

```json
{
  "session_id": "candidate-001",
  "window_start_ms": 1700000000000,
  "window_end_ms": 1700000001000,
  "frame_count": 6,
  "active_violations": {
    "head_pose": {
      "message": "candidate turning away from the screen",
      "severity": 1.0
    },
    "environment_speech": {
      "message": "speech detected in your environment",
      "severity": 0.7
    }
  },
  "mean_face_latency_ms": 45.2,
  "total_latency_ms": 167.8,
  "has_violations": true
}
```

**Violation keywords** in `active_violations`:

| Keyword | Trigger |
|---|---|
| `head_pose` | Head turned left / right / up / down |
| `gaze_position` | Eyes looking away from screen |
| `no_person` | Candidate not visible |
| `multiple_person` | More than one person detected |
| `candidate_speaking` | Candidate's mouth open / speaking |
| `environment_speech` | External voice in audio |
| `environment_noise` | Background device sounds |
| `face_mismatch` | Face does not match registered candidate |
| `background_mismatch` | Exam location changed |
| `object_<class>` | Prohibited object detected (e.g. `object_cell_phone`) |

**Severity** fades from `1.0` (just detected) to `0.0` (cleared) over 10 frames, letting the frontend implement smooth fade-out animations.

---

## Configuration Reference

All settings can be overridden via environment variables or the `.env` file. Key knobs:

| Setting | Default | Description |
|---|---|---|
| `HEAD_POSE_YAW_THRESHOLD` | `20.0` | Yaw angle (degrees) before head-pose violation fires |
| `HEAD_POSE_PITCH_THRESHOLD` | `20.0` | Pitch angle threshold |
| `EYE_GAZE_LEFT_THRESHOLD` | `0.35` | Iris ratio below which gaze-left fires |
| `EYE_GAZE_RIGHT_THRESHOLD` | `0.65` | Iris ratio above which gaze-right fires |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.35` | Minimum YOLO detection confidence |
| `YOLO_PERSON_CONFIDENCE_THRESHOLD` | `0.60` | Stricter threshold for person class |
| `YOLO_FRAME_SKIP` | `6` | Submit every Nth frame to YOLO (~5 FPS at 30 FPS input) |
| `LIP_SYNC_MAR_THRESHOLD` | `0.02` | MAR below which lips are considered still (audio lip-sync suppression) |
| `MOUTH_OPEN_MAR_THRESHOLD` | `0.25` | MAR above which mouth-open violation fires |
| `FACE_SIMILARITY_THRESHOLD` | `0.70` | Cosine similarity threshold for face verification |
| `VERIFICATION_INTERVAL_SECONDS` | `60` | How often periodic face verification runs |
| `BG_SIMILARITY_THRESHOLD` | `0.85` | Cosine similarity threshold for background matching |
| `BG_CHECK_INTERVAL_SECONDS` | `60.0` | Background validation interval |

---

## Logs

All logs are written to `engine_logs/` relative to the project root:

| File | Contents |
|---|---|
| `video_violations.log` | Frame-level violation events |
| `audio_violations.log` | YAMNet audio anomaly events |
| `yolo_detections.log` | Raw YOLO detection results |
| `mediapipe_detections.log` | MediaPipe landmark results |
| `yamnet_detections.log` | YAMNet class scores |
| `background_violations.log` | Background change events |
| `api_sessions.log` | WebSocket connect/disconnect events |
| `face_verification.log` | Identity verification results |
| `errors.log` | Server errors |

Audio segments are saved as WAV files in `audio_chunks/` for debugging.

---

## Development Notes

- **Single worker**: The server runs with one Uvicorn worker by default. All sessions share one process and pipeline instances. Add workers only if RAM permits (each worker loads its own full model set).
- **NumPy version**: Must be `<2.0` for TensorFlow and MediaPipe compatibility.
- **Windows asyncio**: The database connection module automatically sets `WindowsSelectorEventLoopPolicy` for asyncpg compatibility.
- **Model warm-up**: All seven models are loaded at startup before any connection is accepted — no cold-start latency on the first frame.
- **Thread safety**: `InferenceEngine.process_frame()` is single-threaded. YOLO runs in a background thread with a bounded queue. YAMNet runs in its own audio thread. `process_audio()` is thread-safe.

---

## License

[Add your license here]
