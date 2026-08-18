"""Persistent local web launcher for Spec-FastGS.

Run this process once, then select a server-side dataset path and start/stop
training from the browser. Training remains an isolated subprocess, while this
dashboard survives completed and failed runs.
"""

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import uuid
import shutil
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

# Explicit allow-list for advanced options exposed by the browser. Keeping the
# mapping here prevents the UI from turning into an arbitrary command runner.
TRAIN_OPTIONS = {
    "sh_degree": ("--sh_degree", int, 0, 3),
    "asg_degree": ("--asg_degree", int, 1, 128),
    "specular_start_iter": ("--specular_start_iter", int, 0, 1_000_000),
    "position_lr_init": ("--position_lr_init", float, 0.0, 1.0),
    "position_lr_final": ("--position_lr_final", float, 0.0, 1.0),
    "feature_lr": ("--feature_lr", float, 0.0, 1.0),
    "highfeature_lr": ("--highfeature_lr", float, 0.0, 1.0),
    "opacity_lr": ("--opacity_lr", float, 0.0, 1.0),
    "scaling_lr": ("--scaling_lr", float, 0.0, 1.0),
    "rotation_lr": ("--rotation_lr", float, 0.0, 1.0),
    "densify_from_iter": ("--densify_from_iter", int, 0, 1_000_000),
    "densify_until_iter": ("--densify_until_iter", int, 0, 1_000_000),
    "densification_interval": ("--densification_interval", int, 1, 1_000_000),
    "opacity_reset_interval": ("--opacity_reset_interval", int, 1, 1_000_000),
    "densification_refscore_interval": ("--densification_refscore_interval", int, 1, 1_000_000),
    "grad_thresh": ("--grad_thresh", float, 0.0, 1.0),
    "grad_abs_thresh": ("--grad_abs_thresh", float, 0.0, 1.0),
    "loss_thresh": ("--loss_thresh", float, 0.0, 1.0),
    "num_score_cameras": ("--num_score_cameras", int, 1, 10_000),
    "mult": ("--mult", float, 0.0, 10.0),
    "lambda_dssim": ("--lambda_dssim", float, 0.0, 1.0),
    "lambda_spec_l1_weight": ("--lambda_spec_l1_weight", float, 0.0, 100.0),
    "lambda_spec_reg": ("--lambda_spec_reg", float, 0.0, 100.0),
    "sh_spec_min_metric_count": ("--sh_spec_min_metric_count", int, 1, 10_000),
    "sh_spec_mask_threshold": ("--sh_spec_mask_threshold", float, 0.0, 1.0),
    "sh_spec_mask_start": ("--sh_spec_mask_start", int, 0, 1_000_000),
    "sh_spec_grad_scale": ("--sh_spec_grad_scale", float, 0.0, 1.0),
    "adaptive_prior_start": ("--adaptive_prior_start", int, 0, 1_000_000),
    "adaptive_prior_interval": ("--adaptive_prior_interval", int, 1, 1_000_000),
    "adaptive_prior_num_cameras": ("--adaptive_prior_num_cameras", int, 1, 10_000),
    "adaptive_prior_ema": ("--adaptive_prior_ema", float, 0.0, 1.0),
    "refscore_threshold_min": ("--refscore_threshold_min", float, 0.0, 1.0),
    "refscore_threshold_max": ("--refscore_threshold_max", float, 0.0, 1.0),
    "ti_thresh": ("--ti_thresh", float, 0.0, 1.0),
    "ti_bright": ("--ti_bright", float, 0.0, 1.0),
    "sk_intensity": ("--sk_intensity", float, 0.0, 1.0),
    "sk_saturation": ("--sk_saturation", float, 0.0, 1.0),
    "checkpoint_interval": ("--checkpoint_interval", int, 0, 1_000_000),
}
TRAIN_FLAGS = {
    "use_ref_score": "--use_ref_score",
    "use_adaptive_prior": "--use_adaptive_prior",
    "use_sh_spec_mask": "--use_sh_spec_mask",
    "random_background": "--random_background",
    "is_real": "--is_real",
    "is_indoor": "--is_indoor",
}
TRAIN_CHOICES = {
    "ref_prior_method": ("--ref_prior_method", {"tan", "shafer", "hybrid"}),
}
TIMELINE_COMPONENTS = {
    "render", "asg_only", "sh_only", "residual_remaining", "geometry"
}


class LauncherState:
    def __init__(self):
        self.process = None
        self.log = deque(maxlen=120)
        self.preview_root = None
        self.preview_files = {}
        self.uploads = {}
        self.model_path = None
        self.training_started_at = None
        self.target_iterations = 0
        self.source_job = {
            "running": False, "log": deque(maxlen=300), "dataset": None,
            "error": None, "percent": 0.0, "phase": "Idle", "detail": "",
            "started_at": None, "updated_at": None,
        }
        self.lock = threading.Lock()

    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, config):
        with self.lock:
            if self.running():
                raise RuntimeError("A training process is already running")
            dataset = resolve_dataset(config.get("dataset", ""))
            iterations = max(1, int(config.get("iterations", 30000)))
            interval = max(1, int(config.get("stream_interval", 10)))
            images = str(config.get("images", "images"))
            scene = dataset.name
            model = ROOT / "output" / "web_runs" / scene
            self.model_path = model
            model.mkdir(parents=True, exist_ok=True)
            (model / "web_viewer_settings.json").write_text(json.dumps({
                "close_viewer": False,
                "save_frames": bool(config.get("save_frames", False)),
                "record_interval": max(1, int(config.get("record_interval", 50))),
            }))
            command = [
                sys.executable, "-u", "train.py", "-s", str(dataset), "-m", str(model),
                "--iterations", str(iterations), "--web_viewer",
                "--web_http_port", "8081", "--web_ws_port", "6009",
                "--web_stream_interval", str(interval),
            ]
            if images:
                command.extend(["--images", images])
            if config.get("eval", True):
                command.append("--eval")
            if config.get("white_background", False):
                command.append("--white_background")
            advanced = config.get("train_options") or {}
            for name, (flag, cast, minimum, maximum) in TRAIN_OPTIONS.items():
                if name not in advanced or advanced[name] in (None, ""):
                    continue
                value = cast(advanced[name])
                if not minimum <= value <= maximum:
                    raise ValueError("{} must be between {} and {}".format(
                        name, minimum, maximum
                    ))
                command.extend([flag, str(value)])
            for name, flag in TRAIN_FLAGS.items():
                if advanced.get(name) is True:
                    command.append(flag)
            for name, (flag, choices) in TRAIN_CHOICES.items():
                if name not in advanced:
                    continue
                value = str(advanced[name])
                if value not in choices:
                    raise ValueError("Invalid {}: {}".format(name, value))
                command.extend([flag, value])
            checkpoint_iterations = str(advanced.get("checkpoint_iterations", "")).strip()
            if checkpoint_iterations:
                values = [int(value.strip()) for value in checkpoint_iterations.split(",")]
                if any(value < 1 or value > iterations for value in values):
                    raise ValueError("Checkpoint iterations must be within the training range")
                command.extend(["--checkpoint_iterations", ",".join(map(str, values))])
            self.training_started_at = time.time()
            self.target_iterations = iterations
            self.log.clear()
            self.log.append("$ " + " ".join(command))
            self.process = subprocess.Popen(
                command, cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            threading.Thread(target=self._read_log, daemon=True).start()

    def _read_log(self):
        process = self.process
        for line in process.stdout:
            self.log.append(line.rstrip())
        code = process.wait()
        self.log.append("Training finished with exit code {}".format(code))

    def stop(self):
        with self.lock:
            if not self.running():
                return
            self.log.append("Stopping training...")
            self.process.terminate()


STATE = LauncherState()


def resolve_dataset(raw):
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise ValueError("Dataset directory does not exist: {}".format(path))
    if not ((path / "transforms_train.json").is_file()
            or (path / "transforms.json").is_file()
            or (path / "sparse").is_dir()):
        raise ValueError("Folder is not a supported Blender/COLMAP dataset")
    return path


def available_image_sets(root):
    candidates = []
    for path in root.iterdir():
        if path.is_dir() and (path.name == "train" or path.name == "images"
                             or path.name.startswith("images_")):
            count = sum(1 for item in path.rglob("*") if item.is_file()
                        and item.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                        and not item.name.lower().endswith("_normal.png"))
            if count:
                candidates.append({"name": path.name, "count": count})
    return sorted(candidates, key=lambda item: (item["name"] != "images", item["name"]))


def select_preview_set(root, image_set=None):
    sets = available_image_sets(root)
    names = {item["name"] for item in sets}
    if image_set not in names:
        image_set = "train" if "train" in names else (sets[0]["name"] if sets else None)
    image_root = root / image_set if image_set else root
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    files = []
    for path in image_root.rglob("*"):
        lower = path.name.lower()
        if (path.is_file() and path.suffix.lower() in extensions
                and "reflection_prior" not in path.parts
                and not lower.endswith("_normal.png") and "_ref_" not in lower):
            files.append(path)
    files.sort(key=lambda p: p.as_posix())
    STATE.preview_root = root
    STATE.preview_files = {str(i): path for i, path in enumerate(files)}
    return {"image_set": image_set, "images": list(STATE.preview_files), "image_sets": sets}


def index_dataset(raw, image_set=None):
    root = resolve_dataset(raw)
    result = select_preview_set(root, image_set)
    result["dataset"] = str(root)
    return result


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def json_response(self, payload, status=200):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def error_response(self, error, status=400):
        self.json_response({"error": str(error)}, status)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/status":
                running = STATE.running()
                iteration = 0
                phase = "idle"
                if STATE.model_path is not None:
                    telemetry = STATE.model_path / "web_viewer_live" / "telemetry.json"
                    if telemetry.is_file():
                        try:
                            live_payload = json.loads(telemetry.read_text())
                            iteration = int(live_payload.get("iteration", 0))
                            phase = str(live_payload.get("phase", "training"))
                        except (OSError, ValueError, json.JSONDecodeError):
                            pass
                    evaluation = STATE.model_path / "evaluation_status.json"
                    if evaluation.is_file():
                        try:
                            eval_payload = json.loads(evaluation.read_text())
                            if eval_payload.get("status") == "running":
                                phase = "evaluation"
                        except (OSError, json.JSONDecodeError):
                            pass
                target = max(0, int(STATE.target_iterations))
                percent = min(100.0, 100.0 * iteration / target) if target else 0.0
                elapsed = (max(0.0, time.time() - STATE.training_started_at)
                           if STATE.training_started_at else 0.0)
                eta = (elapsed * (target - iteration) / iteration
                       if running and phase == "training" and iteration > 0 else 0.0)
                return self.json_response({
                    "running": running, "returncode": None if running
                    else (STATE.process.returncode if STATE.process else None),
                    "log": list(STATE.log), "iteration": iteration,
                    "target_iterations": target, "percent": round(percent, 2),
                    "elapsed_seconds": round(elapsed), "eta_seconds": round(max(0.0, eta)),
                    "phase": phase,
                })
            if parsed.path == "/api/dataset":
                return self.json_response(index_dataset(parse_qs(parsed.query).get("path", [""])[0]))
            if parsed.path == "/api/image-set":
                query = parse_qs(parsed.query)
                if STATE.preview_root is None:
                    raise ValueError("No dataset loaded")
                return self.json_response(select_preview_set(
                    STATE.preview_root, query.get("name", [""])[0]
                ))
            if parsed.path == "/api/image":
                image_id = parse_qs(parsed.query).get("id", [""])[0]
                path = STATE.preview_files.get(image_id)
                if path is None:
                    raise ValueError("Unknown preview image")
                raw = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                return self.wfile.write(raw)
            if parsed.path == "/api/live":
                if STATE.model_path is None or not STATE.running():
                    return self.json_response({"available": False, "stale": True})
                live = STATE.model_path / "web_viewer_live"
                telemetry = live / "telemetry.json"
                if not telemetry.is_file():
                    return self.json_response({"available": False})
                payload = json.loads(telemetry.read_text())
                payload["available"] = True
                for key in ("rgb", "geometry"):
                    image_path = live / (key + ".jpg")
                    if image_path.is_file():
                        payload[key] = base64.b64encode(image_path.read_bytes()).decode("ascii")
                return self.json_response(payload)
            if parsed.path == "/api/viewer/manifest":
                if STATE.model_path is None:
                    return self.json_response({"available": False})
                manifest = STATE.model_path / "web_viewer_manifest.json"
                if not manifest.is_file():
                    return self.json_response({"available": False})
                payload = json.loads(manifest.read_text())
                payload["available"] = True
                return self.json_response(payload)
            if parsed.path == "/api/timeline":
                component = parse_qs(parsed.query).get("component", ["render"])[0]
                if component not in TIMELINE_COMPONENTS:
                    raise ValueError("Unknown timeline component")
                folder = STATE.model_path / "web_viewer_frames" / component if STATE.model_path else None
                frames = sorted(int(path.stem) for path in folder.glob("*.jpg")
                                if path.stem.isdigit()) if folder and folder.is_dir() else []
                return self.json_response({"component": component, "frames": frames})
            if parsed.path == "/api/timeline/frame":
                query = parse_qs(parsed.query)
                component = query.get("component", ["render"])[0]
                iteration = int(query.get("iteration", ["-1"])[0])
                if component not in TIMELINE_COMPONENTS or STATE.model_path is None:
                    raise ValueError("Unknown timeline frame")
                frame_path = STATE.model_path / "web_viewer_frames" / component / f"{iteration:06d}.jpg"
                if not frame_path.is_file():
                    raise ValueError("Timeline frame does not exist")
                raw = frame_path.read_bytes()
                self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw))); self.end_headers()
                return self.wfile.write(raw)
            if parsed.path == "/api/checkpoints":
                root = STATE.model_path / "point_cloud" if STATE.model_path else None
                values = []
                if root and root.is_dir():
                    for path in root.glob("iteration_*"):
                        try: values.append(int(path.name.split("_", 1)[1]))
                        except ValueError: pass
                return self.json_response({"checkpoints": sorted(set(values))})
            if parsed.path == "/api/evaluation":
                if STATE.model_path is None:
                    return self.json_response({"available": False, "status": "idle"})
                status_path = STATE.model_path / "evaluation_status.json"
                payload = (json.loads(status_path.read_text()) if status_path.is_file()
                           else {"status": "waiting", "phase": "Waiting for training",
                                 "detail": "Metrics run automatically after training."})
                results_path = STATE.model_path / "results.json"
                grouped_path = STATE.model_path / "results_grouped.json"
                payload["available"] = status_path.is_file() or results_path.is_file()
                if results_path.is_file():
                    results = json.loads(results_path.read_text())
                    iteration = int(payload.get("iteration", 0))
                    method = f"ours_{iteration}"
                    if method not in results and results:
                        method = sorted(results)[-1]
                    payload["method"] = method
                    payload["metrics"] = results.get(method, {})
                if grouped_path.is_file() and payload.get("method"):
                    grouped = json.loads(grouped_path.read_text())
                    payload["grouped"] = grouped.get(payload["method"], {})
                if payload.get("method"):
                    gt_dir = STATE.model_path / "test" / payload["method"] / "gt"
                    payload["test_views"] = (sum(1 for path in gt_dir.glob("*.png"))
                                               if gt_dir.is_dir() else 0)
                return self.json_response(payload)
            if parsed.path == "/api/video":
                component = parse_qs(parsed.query).get("component", ["render"])[0]
                if component not in TIMELINE_COMPONENTS or STATE.model_path is None:
                    raise ValueError("Unknown video")
                video = STATE.model_path / "videos" / f"timeline_{component}.mp4"
                if not video.is_file():
                    raise ValueError("Video has not been generated")
                self.send_response(200); self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Disposition", f'attachment; filename="{video.name}"')
                self.send_header("Content-Length", str(video.stat().st_size)); self.end_headers()
                with video.open("rb") as source:
                    shutil.copyfileobj(source, self.wfile, length=1024 * 1024)
                return
            if parsed.path == "/api/source/status":
                job = STATE.source_job
                elapsed = max(0.0, time.time() - job["started_at"]) if job.get("started_at") else 0.0
                percent = float(job.get("percent", 0.0))
                eta = (elapsed * (100.0 - percent) / percent
                       if job["running"] and percent >= 1.0 else 0.0)
                return self.json_response({
                    "running": job["running"], "log": list(job["log"]),
                    "dataset": job["dataset"], "error": job["error"],
                    "percent": round(percent, 1), "phase": job.get("phase", "Idle"),
                    "detail": job.get("detail", ""),
                    "elapsed_seconds": round(elapsed), "eta_seconds": round(eta),
                })
            if parsed.path == "/api/browse":
                raw_path = parse_qs(parsed.query).get("path", [str(ROOT / "datasets")])[0]
                folder = Path(raw_path).expanduser().resolve()
                if not folder.is_dir(): raise ValueError("Directory does not exist")
                return self.json_response({"path": str(folder), "parent": str(folder.parent),
                    "directories": sorted(x.name for x in folder.iterdir() if x.is_dir() and not x.name.startswith('.'))})
            if parsed.path == "/api/choose-directory":
                raw_path = parse_qs(parsed.query).get("initial", [str(ROOT / "datasets")])[0]
                initial = Path(raw_path).expanduser()
                if not initial.is_absolute():
                    initial = ROOT / initial
                initial = initial.resolve()
                if not initial.is_dir():
                    initial = initial.parent
                chooser = shutil.which("zenity")
                command = ([chooser, "--file-selection", "--directory",
                            "--title=Choose dataset output folder",
                            "--filename=" + str(initial) + os.sep] if chooser else None)
                if command is None and shutil.which("kdialog"):
                    command = [shutil.which("kdialog"), "--getexistingdirectory", str(initial),
                               "--title", "Choose dataset output folder"]
                if command is None:
                    raise RuntimeError("No native folder chooser found (install zenity or kdialog)")
                selected = subprocess.run(command, text=True, capture_output=True)
                if selected.returncode != 0:
                    return self.json_response({"cancelled": True})
                folder = selected.stdout.strip()
                if not folder:
                    return self.json_response({"cancelled": True})
                return self.json_response({"cancelled": False, "path": folder})
        except Exception as error:
            return self.error_response(error)
        return super().do_GET()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            parsed = urlparse(self.path)
            if parsed.path == "/api/upload/file":
                query = parse_qs(parsed.query)
                upload_id = query.get("upload", [""])[0]
                relative = query.get("path", [""])[0]
                root = STATE.uploads.get(upload_id)
                if root is None:
                    raise ValueError("Unknown upload session")
                destination = (root / relative).resolve()
                if root != destination and root not in destination.parents:
                    raise ValueError("Unsafe upload path")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as output:
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        output.write(chunk)
                        remaining -= len(chunk)
                return self.json_response({"ok": True})

            payload = json.loads(self.rfile.read(length) or b"{}")
            if parsed.path == "/api/upload/start":
                name = Path(payload.get("name", "dataset")).name or "dataset"
                upload_id = uuid.uuid4().hex
                root = (ROOT / "datasets" / "web_uploads" / (name + "_" + upload_id[:8])).resolve()
                root.mkdir(parents=True, exist_ok=False)
                STATE.uploads[upload_id] = root
                return self.json_response({"upload": upload_id, "path": str(root)})
            if parsed.path == "/api/source/start":
                name = Path(payload.get("name", "source")).name or "source"
                upload_id = uuid.uuid4().hex
                root = (ROOT / "datasets" / "source_uploads" / (name + "_" + upload_id[:8])).resolve()
                root.mkdir(parents=True, exist_ok=False); STATE.uploads[upload_id] = root
                return self.json_response({"upload": upload_id, "path": str(root)})
            if parsed.path == "/api/source/build":
                if STATE.source_job["running"]: raise ValueError("A COLMAP job is already running")
                source = STATE.uploads.get(payload.get("upload"))
                if source is None: raise ValueError("Unknown source upload")
                save_root = Path(payload.get("save_path") or ROOT / "datasets" / "generated").expanduser().resolve()
                save_root.mkdir(parents=True, exist_ok=True)
                destination = save_root / (Path(payload.get("name", "scene")).name or "scene")
                now = time.time()
                job = {"running": True, "log": deque(maxlen=300), "dataset": None,
                       "error": None, "percent": 5.0, "phase": "Starting",
                       "detail": "Launching COLMAP pipeline", "started_at": now,
                       "updated_at": now}
                STATE.source_job = job
                def worker():
                    try:
                        from source_pipeline import build_dataset
                        def update_progress(percent, phase, detail=""):
                            job["percent"] = max(job["percent"], min(100.0, float(percent)))
                            job["phase"] = str(phase)
                            job["detail"] = str(detail)
                            job["updated_at"] = time.time()
                        items = [x for x in source.rglob('*') if x.is_file()]
                        actual_source = items[0] if payload.get("source_type") == "video" else source
                        build_dataset(ROOT, actual_source, destination, payload.get("source_type", "images"),
                                      int(payload.get("frame_count", 200)), payload.get("scales", []),
                                      job["log"].append, update_progress)
                        job["dataset"] = str(destination)
                    except Exception as error:
                        job["error"] = str(error); job["phase"] = "Failed"
                        job["detail"] = str(error); job["log"].append("ERROR: " + str(error))
                    finally: job["running"] = False
                threading.Thread(target=worker, daemon=True).start()
                return self.json_response({"ok": True, "destination": str(destination)})
            if parsed.path == "/api/upload/finish":
                root = STATE.uploads.get(payload.get("upload"))
                if root is None:
                    raise ValueError("Unknown upload session")
                result = index_dataset(str(root))
                result["upload"] = payload.get("upload")
                return self.json_response(result)
            if parsed.path == "/api/upload/cancel":
                upload_id = payload.get("upload")
                root = STATE.uploads.pop(upload_id, None)
                if root is not None and root.is_dir():
                    uploads_root = (ROOT / "datasets" / "web_uploads").resolve()
                    if uploads_root in root.parents:
                        shutil.rmtree(str(root))
                return self.json_response({"ok": True})
            if parsed.path == "/api/settings":
                if STATE.model_path is None:
                    raise ValueError("No active viewer model")
                STATE.model_path.mkdir(parents=True, exist_ok=True)
                target = STATE.model_path / "web_viewer_settings.json"
                temporary = target.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload))
                temporary.replace(target)
                return self.json_response({"ok": True})
            if parsed.path == "/api/run":
                STATE.start(payload)
                return self.json_response({"ok": True})
            if parsed.path == "/api/video/generate":
                if STATE.model_path is None:
                    raise ValueError("No viewer run selected")
                component = str(payload.get("component", "render"))
                fps = max(1, min(60, int(payload.get("fps", 30))))
                if component not in TIMELINE_COMPONENTS:
                    raise ValueError("Unknown timeline component")
                frames_dir = STATE.model_path / "web_viewer_frames" / component
                frames = sorted(frames_dir.glob("*.jpg")) if frames_dir.is_dir() else []
                if not frames:
                    raise ValueError("No recorded frames for this component")
                ffmpeg = shutil.which("ffmpeg")
                if not ffmpeg:
                    raise RuntimeError("FFmpeg is not installed")
                videos = STATE.model_path / "videos"; videos.mkdir(parents=True, exist_ok=True)
                manifest = videos / f"timeline_{component}.txt"
                duration = 1.0 / fps
                lines = []
                for frame in frames:
                    lines.extend(["file '{}'".format(str(frame).replace("'", "'\\''")),
                                  "duration {:.9f}".format(duration)])
                lines.append("file '{}'".format(str(frames[-1]).replace("'", "'\\''")))
                manifest.write_text("\n".join(lines) + "\n")
                video = videos / f"timeline_{component}.mp4"
                result = subprocess.run([
                    ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(manifest),
                    "-vf", "format=yuv420p", "-c:v", "libx264", "-movflags", "+faststart",
                    str(video)
                ], capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError("FFmpeg failed: " + result.stderr[-1000:])
                return self.json_response({"ok": True, "frames": len(frames),
                                           "download": f"/api/video?component={component}"})
            if parsed.path == "/api/stop":
                STATE.stop()
                return self.json_response({"ok": True})
            return self.error_response("Unknown API", 404)
        except Exception as error:
            return self.error_response(error)


def main():
    parser = argparse.ArgumentParser(description="Spec-FastGS persistent web launcher")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("Spec-FastGS launcher: http://{}:{}".format(args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        STATE.stop()
        server.server_close()


if __name__ == "__main__":
    main()
