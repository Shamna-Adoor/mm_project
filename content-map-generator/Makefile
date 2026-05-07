.PHONY: setup download-samples analyze eval player label test lint help

PYTHON   := python3
VENV     := .venv
PIP      := $(VENV)/bin/pip
PY       := $(VENV)/bin/python

# ── Setup ─────────────────────────────────────────────────────────────────────

setup: $(VENV)/bin/activate
	@echo "✓ Virtual environment ready. Run: source .venv/bin/activate"

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools
	$(PIP) install --no-build-isolation -r requirements.txt
	@touch $(VENV)/bin/activate

# ── Data ──────────────────────────────────────────────────────────────────────

download-samples:
	$(PY) scripts/download_sample_videos.py

download-samples-force:
	$(PY) scripts/download_sample_videos.py --force

# ── Analysis ──────────────────────────────────────────────────────────────────
# Usage: make analyze VIDEO=data/videos/my_video.mp4
# Optional: WHISPER_MODEL=small FORCE=--force SKIP_VISUAL=--skip-visual

VIDEO        ?=
WHISPER_MODEL ?= base
FORCE        ?=
SKIP_VISUAL  ?=

analyze:
ifndef VIDEO
	$(error VIDEO is required. Usage: make analyze VIDEO=path/to/video.mp4)
endif
	$(PY) -m analyzer.pipeline $(VIDEO) \
		--whisper-model $(WHISPER_MODEL) \
		$(FORCE) $(SKIP_VISUAL)

# ── Evaluation ────────────────────────────────────────────────────────────────
# Usage: make eval VIDEO_ID=lex_fridman_ep400

VIDEO_ID ?=

eval:
ifndef VIDEO_ID
	$(error VIDEO_ID is required. Usage: make eval VIDEO_ID=lex_fridman_ep400)
endif
	$(PY) -m evaluation.compare \
		--gt  data/ground_truth/$(VIDEO_ID).json \
		--pred data/predictions/$(VIDEO_ID).json

# ── Frontend + API server ─────────────────────────────────────────────────────
# `make player` starts both the FastAPI backend (port 8000) and Vite dev server
# (port 5173) in parallel. Ctrl-C kills both.

player:
	cd player && npm install --silent
	bash scripts/start_player.sh

# Build React, then launch as a native desktop app via Electron
app:
	cd player && npm install --silent && npm run build
	cd desktop && npm install --silent && npm run start

# Package into a distributable .app (macOS) or .exe (Windows)
dist:
	cd player && npm install --silent && npm run build
	cd desktop && npm install --silent && npm run dist

server:
	$(PY) -m uvicorn server:app --port 8000 --reload

label:
	cd labeling_tool && npm install && npm run dev

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	$(VENV)/bin/pytest tests/ -v --tb=short

test-cov:
	$(VENV)/bin/pytest tests/ -v --tb=short --cov=analyzer --cov=evaluation --cov-report=term-missing

# ── Lint ──────────────────────────────────────────────────────────────────────

lint:
	$(VENV)/bin/python -m py_compile \
		analyzer/_logging.py \
		analyzer/audio/*.py \
		analyzer/visual/*.py \
		analyzer/fusion.py \
		analyzer/pipeline.py \
		evaluation/metrics.py \
		evaluation/compare.py \
		scripts/download_sample_videos.py
	@echo "✓ No syntax errors"

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  make setup              Create .venv and install Python deps"
	@echo "  make download-samples   Download all videos in scripts/videos.yaml"
	@echo "  make analyze VIDEO=...  Run full pipeline on a video file"
	@echo "  make eval VIDEO_ID=...  Compare GT vs predictions and show metrics"
	@echo "  make player             Start the React video player (dev server)"
	@echo "  make label              Start the React labeling tool (dev server)"
	@echo "  make test               Run pytest suite"
	@echo "  make test-cov           Run pytest with coverage report"
	@echo "  make lint               Syntax-check all Python modules"
	@echo ""
