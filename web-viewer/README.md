# Spec-FastGS Web Viewer

## Recommended: persistent launcher

Activate the same Conda environment used by Spec-FastGS, then run only:

```bash
python web-viewer/launcher.py
```

The **Upload source → COLMAP** panel accepts one video (with a requested output
frame count) or multiple images, a dataset name, destination directory, and
selected downscale factors. It uses the locally built CPU COLMAP executable at
`colmap/build/src/colmap/exe/colmap` and FFmpeg, producing the standard
`images`, `sparse/0`, `images_2`, `images_4`, `images_8` layout. The generated
dataset is selected automatically for training when reconstruction completes.

Open `http://127.0.0.1:8080`. Click **Upload dataset folder**, choose the complete
Blender/COLMAP dataset in File Explorer, preview its input sequence, set training
options, and press **Run training**. The browser preserves and uploads the folder
structure to `datasets/web_uploads/`; the launcher starts `train.py` as a child
process and remains available after training finishes.

Synthetic datasets may use either the standard `transforms_train.json` plus
`transforms_test.json` layout, or a single `transforms.json` with an `images/`
folder. When evaluation is enabled for the single-file layout, every eighth
camera is held out for testing, matching the COLMAP evaluation split.

Use **Run settings…** to configure the representation, learning rates,
densification schedule, FastGS loss controls, and optional reflection guidance
before starting a run. The Predicted RGB card can switch live between the full
render, ASG-only contribution, SH-only contribution, and remaining residual.
The residual uses the selected dataset camera's ground truth; reset orbit for a
pose-aligned comparison.

Enable **Record timeline components** to capture Render, ASG-only, SH/diffuse,
residual, and geometry frames at the configured recording interval. The
timeline dialog can scrub or play those frames and export the selected component
to MP4 through FFmpeg. Checkpoint interval and explicit checkpoint iterations
are configured under **Run settings…**; after training, choose a saved model in
**Interactive checkpoint** to load its real Gaussian, ASG, and specular state
and orbit it in the final viewer.

When an evaluation split is available, training automatically renders all test
views before entering the final interactive viewer, computes PSNR, SSIM, LPIPS,
FPS, and specular diagnostics, and publishes them to the Evaluation metrics card.
Runs without test cameras are marked as skipped instead of producing empty
metric output.

Each launcher start begins with a clean session and does not automatically load
metrics or live telemetry from an older run. The sidebar keeps dataset controls
open and groups COLMAP, playback/checkpoints, camera, and Gaussian display tools
into collapsible sections. Training progress above the observatory reports the
current iteration, percentage, elapsed time, and ETA from the active launcher
process.

For COLMAP/Mip-NeRF 360 scenes, the **Training images** dropdown is populated
from the uploaded folders (`images`, `images_2`, `images_4`, `images_8`, etc.)
and the selection is passed to `train.py --images`. Uploads can be cancelled
mid-transfer; partial server files are removed. The input-loop slider supports
1–60 FPS and only affects preview playback, not training.

## Embedded mode

The viewer can also be started inside `train.py`, keeping all CUDA rendering on the
training thread. Open `http://127.0.0.1:8080` after launching training with
`--web_viewer`.

```bash
python train.py -s datasets/Ref-NeRF/refnerf/toaster -m output/toaster \
  --eval --white_background --web_viewer
```

Use `--web_stream_interval 1` to publish every iteration. This is expensive;
the default is every 10 iterations. `--web_save_frames` stores streamed frames
under `<model_path>/web_viewer_frames`.

When using a non-default websocket port, open the UI with for example
`http://127.0.0.1:8080/?wsPort=6010`.
