# Basketball 16:9 → 9:16 Auto-Crop Pipeline

Automated portrait cropping for basketball footage using object detection, multi-object tracking, and signal-smoothed crop windows.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and FFmpeg on PATH.

## Project Structure

```
stages/
  ingest.py          # Stage 1: Video decode + frame loading
  detection.py       # Stage 2: YOLOv8 per-frame detection (Person 1)
  tracking.py        # Stage 3: ByteTrack multi-object tracking (Person 1)
  focus.py           # Stage 4: Focus point computation (Person 2)
  smoothing.py       # Stage 5: Crop window smoothing (Person 2)
  encode.py          # Stage 6: Crop + encode to portrait MP4 (Person 2)
  eval_harness.py    # Stage 7: Scoring + parameter sweep (Person 3)
main_pipeline.py     # Integration — wires all stages together
run_sweep.py         # Hyperparameter sweep runner
viewer/index.html    # Side-by-side clip viewer
clips/               # Source MP4s (not committed)
output/              # Generated portraits + sweep results
```

## Usage

### Single clip
```bash
python main_pipeline.py clips/game1.mp4 -o output/game1_portrait.mp4
```

### With custom parameters
```bash
python main_pipeline.py clips/game1.mp4 --sigma 12 --lookahead 0.3 --ball-conf 0.4
```

### Parameter sweep
```bash
python run_sweep.py clips/game1.mp4
python run_sweep.py clips/ --top 5
```

### Viewer
Open `viewer/index.html` in a browser. Edit the `CLIPS` array in the HTML to point at your clip pairs.

## Division of Labor

| Person | Stages | Files |
|--------|--------|-------|
| Person 1 (you) | Detection + Tracking | `detection.py`, `tracking.py` |
| Person 2 | Focus + Smoothing + Encode | `focus.py`, `smoothing.py`, `encode.py` |
| Person 3 | Ingest + Eval Harness | `ingest.py`, `eval_harness.py`, `run_sweep.py` |
| Person 4 | Viewer | `viewer/index.html` |
