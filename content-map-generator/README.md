# content-map-generator

Automatically detect and skip non-content segments (intros, sponsor reads, outros, dead air) in long-form YouTube videos. Drop in a video, get a JSON segment map, load it in the custom player — done.

---

## Architecture

```mermaid
flowchart TD
    VIDEO[Video .mp4] --> AE[extract_audio\n16 kHz WAV]
    VIDEO --> EF[extract_frames\n1 fps JPEGs]

    AE --> TR[transcribe\nWhisper]
    AE --> SD[silence_detect\npydub]
    AE --> MD[music_detect\nlibrosa]
    TR --> TS[transcript_signals\nsponsors · boilerplate]

    EF --> SC[scene_detect\nPySceneDetect]
    EF --> OCR[ocr_frames\npytesseract]
    EF --> MA[motion_analysis\nOpenCV diff]

    SD & MD & TS --> AS[Audio Signals JSON]
    SC & OCR & MA --> VS[Visual Signals JSON]

    AS & VS --> FU[fusion.py\nrule-based classifier]
    FU --> SEG[Segment Map JSON]

    SEG --> PL[React Player\nauto-skip UI]
    SEG --> LT[Labeling Tool\nhand-label + export]
    SEG --> EV[Evaluation\nIoU · F1 · skip accuracy]
```

---

## Setup

### System dependencies

**macOS**
```bash
brew install tesseract ffmpeg
```

**Linux (Debian/Ubuntu)**
```bash
sudo apt install tesseract-ocr ffmpeg
```

### Python environment

```bash
make setup
source .venv/bin/activate
```

---

## End-to-end example

```bash
# 1. Download a sample video
make download-samples

# 2. Run the full pipeline
make analyze VIDEO=data/videos/lex_fridman_ep400.mp4

# Output: data/predictions/lex_fridman_ep400.json

# 3. Open the player
make player
# → http://localhost:5173
# Load a local file, or paste a YouTube/Twitch/direct media URL in the URL / Stream control.
# Finite VOD links are downloaded + segmented; live HLS streams are played stream-only.

# 4. (Optional) Hand-label a video
make label
# → http://localhost:5174

# 5. Evaluate auto-predictions against a hand-labeled ground truth
make eval VIDEO_ID=lex_fridman_ep400
```

---

## Player screenshot

> *Screenshot placeholder — add after Phase 2 is complete.*

---

## Segment JSON format

```json
{
  "video_id": "lex_fridman_ep400",
  "duration_seconds": 9240,
  "generated_at": "2024-05-01T12:00:00Z",
  "segments": [
    {
      "start": 0,
      "end": 42,
      "label": "intro",
      "confidence": 0.91,
      "skip_recommended": true,
      "reason": "Music + boilerplate greeting near video start",
      "signals_used": ["music_intervals", "boilerplate_phrases"]
    },
    {
      "start": 42,
      "end": 9180,
      "label": "main_content",
      "confidence": 0.6,
      "skip_recommended": false,
      "reason": "Default classification",
      "signals_used": []
    }
  ]
}
```

Full schema: [`schemas/segment_schema.json`](schemas/segment_schema.json)

---

## Project structure

```
content-map-generator/
├── analyzer/
│   ├── audio/          # extract_audio, transcribe, silence, music, transcript_signals
│   ├── visual/         # extract_frames, scene_detect, ocr_frames, motion_analysis
│   ├── fusion.py       # rule-based segment classifier
│   ├── fusion_rules.yaml
│   └── pipeline.py     # top-level orchestrator
├── player/             # React + Vite + TypeScript player
├── labeling_tool/      # React labeling UI
├── evaluation/         # metrics.py, compare.py
├── schemas/            # segment_schema.json, signal_schema.json
├── scripts/            # download_sample_videos.py, videos.yaml
├── tests/
├── data/
│   ├── videos/         # raw .mp4 files (gitignored)
│   ├── intermediate/   # cached audio, frames, transcripts (gitignored)
│   ├── ground_truth/   # hand-labeled JSON
│   └── predictions/    # auto-generated JSON
└── Makefile
```

---

## Make targets

| Target | Description |
|---|---|
| `make setup` | Create `.venv` and install Python deps |
| `make download-samples` | Download all videos in `scripts/videos.yaml` |
| `make analyze VIDEO=path` | Run the full pipeline on a video |
| `make eval VIDEO_ID=name` | Compare GT vs predictions, show metrics |
| `make player` | Start the React player dev server |
| `make label` | Start the labeling tool dev server |
| `make test` | Run pytest suite |
| `make lint` | Syntax-check all Python modules |

---

## URL and stream import

The player can import:

- Local media files (`.mp4`, `.mov`, `.mkv`, `.webm`, etc.)
- YouTube videos and VODs supported by `yt-dlp`
- Twitch VODs and live streams supported by `yt-dlp`
- Direct media URLs, including `.mp4` and `.m3u8` HLS playlists

Backend endpoint:

```bash
curl -X POST http://localhost:8000/api/import-url \
  -F "source_url=https://www.youtube.com/watch?v=..." \
  -F "analyze=true"
```

Finite videos are downloaded into `data/uploads/`, analyzed, and served back via
`/api/video/{job_id}`. Live streams and stream-only HLS links are playable, but
offline segmentation is not queued because there is no finite duration to segment.
