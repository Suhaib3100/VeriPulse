# VeriPulse — Real-Time Deepfake Vishing Defense

> **This is not a "deepfake detector app". This is a real-time trust enforcement layer for human communication.**

## Quick Start

### 1. Installation
```bash
# Create virtual environment
python -m venv .venv
# Activate it (Windows)
.venv\Scripts\Activate
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Liveness Demo
Test the physiological liveness detection (rPPG) with your webcam:
```bash
python run_veripulse_demo.py
```

### 3. Run Backend API
```bash
cd apps/backend
uvicorn main:app --reload
```

### 4. Frontend
```bash
cd apps/web
npm run dev
```

## Features
- **Physiological Liveness**: Extracts heart rate (rPPG) from facial video to detect synthetic faces.
- **Active Challenges**: Validates user response to random prompts (blinks, head turns).
- **Trust Scoring**: Fuses multiple signals into a single trust score.

## Structure

- `core/`
  - `liveness/` - Liveness logic (Physiological & Active)
  - `rppg/` - Signal extraction (POS algorithm) & filtering
  - `vision/` - Face detection & tracking
  - `scoring/` - Trust scoring models
  - `policy/` - Security policies
- `apps/backend/` - FastAPI server
- `apps/web/` - Next.js frontend
- `scripts/` - Utility scripts
- `run_veripulse_demo.py` - Main demo entry point

## Trust States

| Score | State | Description |
|-------|-------|-------------|
| ≥ 0.7 | ✅ Verified | Consistent physiological signals & passed challenges |
| 0.4–0.7 | ⚠️ Suspicious | Inconsistent signals or missing data |
| < 0.4 | 🚫 Likely Synthetic | No pulse detected or failed challenges |

MIT License
