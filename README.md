# VeriPulse — Real-Time Deepfake Vishing Defense

VeriPulse is a real-time “reality firewall” for video calls that detects deepfake impersonation using physiological liveness signals (rPPG) plus challenge-response.  
It helps prevent AI vishing scams (family fraud, CEO fraud, fake KYC) by providing an on-screen trust score and blocking risky actions when authenticity is uncertain.

## Problem

In 2025, attackers can clone voice/video in real time and impersonate trusted people during calls. This breaks the human trust layer: if a scammer looks and sounds like someone you trust, traditional security controls (passwords/MFA) don’t help the victim in that moment.

## Our Solution

A layered verifier that focuses on *liveness* and *intent-to-fraud signals*:

1. **Physiological liveness (rPPG)**
   - Extract subtle blood-volume pulse signals from facial regions in the video stream.
   - Output a confidence score: “does this video contain a plausible live pulse signal?”

2. **Active liveness challenge (challenge-response)**
   - If confidence is low/medium, trigger quick challenges (head turn, blink sequence).
   - Validate timing + face geometry consistency.

3. **Trust Overlay + Policy**
   - UI overlay shows: Verified / Suspicious / Likely Synthetic.
   - Optional policy mode: block “high-risk actions” (share OTP, approve transfer) unless verified.

## Demo (what we will show)

- A live real webcam feed → overlay shows **Verified** + stable pulse.
- A deepfake/face-swap video (pre-recorded for safety) → overlay shows **Likely Synthetic** / unstable pulse / failed challenge.
- A “risky action” button (wire transfer / share OTP) → disabled unless Verified.

## Architecture (MVP)

- **Capture**: Webcam/WebRTC stream
- **Detection Engine**: Face ROI + stabilization → rPPG extraction → feature scoring → classifier
- **Challenge Engine**: prompts + motion verification
- **Overlay UI**: Trust score + state + actions
- **(Optional)**: Logging + telemetry dashboard

## Tech Stack (prototype)

- Python 3.11+
- OpenCV (face ROI, stabilization, signal extraction)
- (Optional) MediaPipe Face Mesh (robust landmarks)
- Simple classifier: Logistic Regression / XGBoost (fast)
- Frontend overlay: React + Vite (or a simple HTML canvas overlay)
- Docker for reproducibility

## Repo Structure

.
├── apps/
│   ├── demo_web/                 # Web demo (WebRTC + overlay UI)
│   └── demo_desktop/             # Desktop demo (optional)
├── veripulse/
│   ├── vision/                   # face detection + ROI tracking
│   ├── rppg/                     # pulse extraction + features
│   ├── liveness/                 # challenge-response verification
│   ├── scoring/                  # risk scoring + thresholds
│   └── utils/
├── scripts/
│   ├── run_webcam_demo.py
│   ├── run_video_demo.py
│   └── generate_report.py
├── assets/
│   ├── sample_real.mp4
│   └── sample_fake.mp4
├── docs/
│   ├── architecture.md
│   └── demo_script.md
├── requirements.txt
├── Dockerfile
└── README.md

## Quickstart (placeholder)

> Coming soon — repo is being initialized for the hackathon build.

Planned:
1) `pip install -r requirements.txt`  
2) `python scripts/run_webcam_demo.py`

## Safety & Ethics

- Demo uses consented videos only.
- No deepfake generation is included in this repo.
- Output is a risk indicator, not absolute proof.

## License

MIT
