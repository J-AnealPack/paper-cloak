# Paper Cloak

Tracks a flat page, hides objects covering it, and updates newly exposed writing.

## Setup

Run commands from the folder containing `requirements.txt` and `run_test.py`.

```powershell
python -m pip install -r requirements.txt
```

## Run

**Webcam, with perspective tracking and writing updates:**

```powershell
python -m paper_cloak.app --source 0 --model homography --dynamic
```

Try `--source 1` if the wrong camera opens. Press **Space**, click the page corners **TL → TR → BR → BL**, then **Enter**. Start with an unobstructed, flat page containing printed details.

**Video test menu:**

```powershell
python run_test.py
```

Or choose directly: `python run_test.py 3`.

| Number | Test |
|---|---|
| 1 | Moving hand |
| 2 | Perspective changes |
| 3 | Hand + perspective + new writing |
| 4 | Heavy-occlusion stress test |
| 5 | Original demo, if `demo/` is retained |

**Recreate deleted test cases:**

```powershell
python generate_tests.py --output test_cases
```

Add `--case writing` to generate only test 3. Generation saves videos and results; normal playback does not.

**Optional development checks:**

```powershell
python -m unittest discover -s tests -v
```

**Original demo generator, if retained:** `python -m paper_cloak.demo` creates the separate `demo/` files.

## Controls

**D:** toggle updates · **R:** reselect a clean page · **S:** save clean-page image · **Q/Esc:** quit.

Panels: raw input, processed output, saved page, and blocked update regions. Keep page details visible; if tracking is lost, uncover the page and press **R**.

## Python files

| File | Purpose |
|---|---|
| `run_test.py` | Launch an existing video test. |
| `generate_tests.py` | Create the synthetic test videos and saved results. |
| `paper_cloak/app.py` | Camera/video input, corner selection, display and controls. |
| `paper_cloak/pipeline.py` | Connect the processing steps. |
| `paper_cloak/tracking.py` | Follow page features and estimate movement. |
| `paper_cloak/geometry.py` | Normalize the page and overlay it onto the video. |
| `paper_cloak/occlusion.py` | Identify regions that should not update. |
| `paper_cloak/memory.py` | Preserve covered content and learn exposed content. |
| `paper_cloak/config.py` | Default settings and config loading. |
| `paper_cloak/__init__.py` | Expose the package's main classes. |
| `paper_cloak/__main__.py` | Enable `python -m paper_cloak`; optional with the commands above. |
| `paper_cloak/demo.py` | Generate the original, simpler demo. Optional. |
| `tests/test_core.py` | Automated processing checks. Optional. |

Each retained test needs its input video, `config.json` and `corners.json`. Other generated images, comparisons and metrics are optional. Dynamic updates are designed for white paper and dark ink.
