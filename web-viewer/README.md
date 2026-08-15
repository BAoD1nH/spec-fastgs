# Spec-FastGS Web Viewer

The viewer is started inside `train.py`, keeping all CUDA rendering on the
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
