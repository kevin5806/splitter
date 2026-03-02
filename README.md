# Image Splitter and Resizer
![python_sSZbYKqzIa](https://github.com/user-attachments/assets/7c70a711-db13-456a-8772-0f40cc45545e)

Batch image splitter with per-image settings, smart scan-aware grid detection, crop tracking overlays, and short crop IDs (`A1`, `A2`, ...).

---

## Quick Start (Windows EXE)

Build executable:

```bat
build_exe.bat
```

Run app:

```text
dist\Splitter.exe
```

---

## Main Features

- Per-image overrides (grid, folder, smart grid, crop margin)
- Smart grid detection for imperfect scanned sheets
- Smart grid supports non-uniform block sizes (different cell widths/heights)
- Clear output quality control (`Massima`, `Bilanciata`, `Compatta`)
- Orientation selector (`Auto`, `Orizzontale`, `Verticale`) global and per-image
- Configurable crop margin in pixels (global, per-image, or CLI)
- Automatic crop skew control to reduce tilted cut results
- Post-conversion report window with per-file status, score, method, and diagnostics summary
- Crop filenames with short IDs (`A1`, `B3`, ...)
- Automatic detection of impossible files at process start
- Impossible files copied to `_impossible_files/<timestamp>` with CSV report
- Tracking files for each source sheet:
  - `<source>_tracking_overlay.jpg` (boxes + labels on original sheet)
  - `<source>_tracking_index.csv` (code-to-coordinate mapping)
  - `<source>_diagnostics.json` (recognition quality, inferred lines, malformed/offscreen stats)
- Selectable base output location from file explorer
- CLI mode and GUI mode

---

## Output Behavior

For a 4x4 sheet, crops are named:

```text
A1.jpg A2.jpg ... A4.jpg
B1.jpg ... D4.jpg
```

Additional files in the same output folder:

```text
<source>_tracking_overlay.jpg
<source>_tracking_index.csv
<source>_diagnostics.json
```

---

## Project Files

| File | Purpose |
|------|---------|
| `src/` | Real source package (`splitter_with_per_image`, `splitter_core`, `splitter_models`) |
| `splitter_with_per_image.py` | Compatibility wrapper + executable entrypoint |
| `splitter_core.py` | Compatibility wrapper for core module |
| `splitter_models.py` | Compatibility wrapper for models module |
| `requirements.txt` | Runtime dependencies |
| `requirements-dev.txt` | Runtime + test dependencies |
| `run_tests.bat` | Run tests on Windows |
| `build_exe.bat` | Build standalone Windows EXE |
| `tooltips.json` | UI tooltip text |
| `.gitignore` | Ignore cache/build/output artifacts for a cleaner workspace |

Notes:
- Legacy Linux run scripts are intentionally removed.
- Keep datasets outside the repository (recommended path on this workstation: `C:\GitHub\splitter-local-data\AllSamples`).
- Do not commit local scans, test exports, or diagnostics JSON generated from real customer data.

---

## Requirements (Development)

- Python 3.8+ (required only for development/build/tests)

Runtime dependencies for source build:

- Pillow
- tkinterdnd2
- sv-ttk
- numpy
- opencv-python-headless

Test dependencies (via `requirements-dev.txt`):

- pytest
- pytest-cov

---

## Usage Overview

1. Build and launch `dist\Splitter.exe`
2. Load images (browse or drag and drop)
3. Set global defaults:
   - Output quality (choose **Massima** for best detail)
   - Across / High
   - Crop margin (px)
   - Output folder
   - Output location (browse selectable)
4. Optionally set per-image overrides
5. Process all images
6. If impossible files are detected, they are skipped and copied to a dedicated folder

---

## Unit Tests

Windows:

```bat
run_tests.bat
```

Direct:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -v --cov=src.splitter_with_per_image --cov=src.splitter_core --cov=src.splitter_models --cov-report=term-missing tests
```

CLI example with margin and forced orientation:

```bash
python splitter_with_per_image.py sheet.jpg --quality max --orientation horizontal --across 4 --high 4 --crop-margin 12
```

---

## Build EXE (Windows)

```bat
build_exe.bat
```

Produces:

```text
dist\Splitter.exe
```

---

## License

MIT (see `LICENSE`)
