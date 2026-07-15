@echo off
rem One-shot wrapper for the resumable 2024_train cache build.
rem Launched via Task Scheduler so it is NOT a child of any editor/agent
rem session (two earlier attempts died when the session that spawned them
rem closed). Appends to build_2024_train.log in the repo root; safe to re-run
rem any time — the builder resumes from the last completed day.
set PYTHONUTF8=1
cd /d D:\UWA\Honours\Honours-Project
.venv\Scripts\python.exe build_period_cache_lowmem.py --period 2024_train >> build_2024_train.log 2>&1
