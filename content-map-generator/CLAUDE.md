# content-map-generator — CLAUDE.md

## Project purpose
Analyse long-form YouTube videos (podcasts, lectures, interviews) and produce
a JSON metadata file that a custom React video player consumes to auto-skip
non-content segments (intros, sponsor reads, outros, dead air).

## Build order
Phases are strictly ordered. Do NOT start the next phase until the user
confirms the previous one.

1. **Phase 1** — Scaffolding + schemas *(done)*
2. **Phase 2** — React player + labeling tool *(next)*
3. **Phase 3** — Audio analysis (Whisper, silence, music, transcript signals)
4. **Phase 4** — Visual analysis (frames, scenes, OCR, motion)
5. **Phase 5** — Fusion (rule-based, weights in fusion_rules.yaml)
6. **Phase 6** — Evaluation metrics + visualisation

## Key design decisions
- Player is built *before* the analyzer — a hand-labeled JSON produces a full
  demo even if auto-analysis is never finished.
- Fusion is rule-based only (no ML training). Weights live in
  `analyzer/fusion_rules.yaml`.
- All intermediate outputs are cached in `data/intermediate/`. Slow stages
  (Whisper, OCR) must check the cache first and skip unless `--force`.
- Logging: use `analyzer._logging.get_logger(__name__)` everywhere. No bare
  `print()` in library code.

## Schemas
`schemas/segment_schema.json` and `schemas/signal_schema.json` are the
contract between every component. Do not change field names or types without
updating both schemas, all callers, and the tests.

## Running locally
```bash
# First-time setup
brew install tesseract ffmpeg   # macOS system deps
make setup                      # creates .venv, installs Python deps

# Download sample videos
make download-samples

# Run analysis
make analyze VIDEO=data/videos/lex_fridman_ep400.mp4

# Start the player
make player
```

## Tests
```bash
make test
```
Tests live in `tests/`. Phase 1 tests validate schemas and confirm that stubs
raise `NotImplementedError` (expected). Tests grow each phase.
