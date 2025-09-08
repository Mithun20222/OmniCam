# OmniCam - Face Recognition System

OmniCam is a **real-time face recognition system** built with [DeepFace](https://github.com/serengil/deepface) and OpenCV.  
It recognizes authorized users, detects intruders, and keeps track of their status using a flag system.

## ✨ Features
- 🚀 Fast recognition with **SFace** model (optimized for CPU).
- ✅ Tracks authorized users with `flag=1`.
- 🚨 Detects intruders with `flag=-1` (stable, requires multiple frames before confirmation).
- 🎥 Real-time detection from webcam.
- 🟢 Stable labels with smoothing (no flickering).
- 🔒 Authorized face embeddings loaded from `data/authorized/`.
