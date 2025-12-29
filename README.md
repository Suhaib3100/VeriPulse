# VeriPulse — Real-Time Deepfake Vishing Defense

> **This is not a "deepfake detector app". This is a real-time trust enforcement layer for human communication.**

## Quick Start

```bash
# Backend
pip install -r requirements.txt
cd apps/backend && uvicorn main:app --reload

# Frontend (after Next.js init in apps/web)
cd apps/web && npm run dev
```

## Structure

- `core/` - Product logic (vision, rPPG, liveness, scoring, policy)
- `apps/backend/` - FastAPI server
- `apps/web/` - Next.js frontend (initialize separately)
- `scripts/` - Demo runners
- `docs/` - Architecture docs

## Trust States

| Score | State |
|-------|-------|
| ≥ 0.75 | ✅ Verified |
| 0.4–0.75 | ⚠️ Suspicious |
| < 0.4 | 🚫 Likely Synthetic |

MIT License
