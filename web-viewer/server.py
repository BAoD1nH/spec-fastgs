"""Live browser viewer bridge for Spec-FastGS training.

The websocket thread never touches CUDA.  Training owns rendering and calls
``poll_settings``/``publish`` at safe points between optimization iterations.
"""

import asyncio
import base64
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import websockets


class ViewerServer:
    def __init__(self, host="127.0.0.1", http_port=8080, ws_port=6009):
        self.host = host
        self.http_port = int(http_port)
        self.ws_port = int(ws_port)
        self.static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        self._lock = threading.Lock()
        self._settings = {
            "enabled": True,
            "paused": False,
            "interval": 10,
            "yaw": 0.0,
            "pitch": 0.0,
            "zoom": 1.0,
            "fov_scale": 1.2,
            "splat_scale": 1.35,
            "geometry_opacity": 0.72,
            "save_frames": False,
        }
        self._clients = set()
        self._loop = None

    def start(self):
        handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(
            *args, directory=self.static_dir, **kwargs
        )
        httpd = ThreadingHTTPServer((self.host, self.http_port), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        threading.Thread(target=self._run_ws, daemon=True).start()
        print(f"Web viewer: http://{self.host}:{self.http_port}")

    def configure(self, **settings):
        with self._lock:
            self._settings.update({k: v for k, v in settings.items() if k in self._settings})

    def _run_ws(self):
        asyncio.run(self._serve_ws())

    async def _serve_ws(self):
        self._loop = asyncio.get_running_loop()
        async with websockets.serve(self._client, self.host, self.ws_port, max_size=2**22):
            await asyncio.Future()

    async def _client(self, websocket, path=None):
        self._clients.add(websocket)
        try:
            await websocket.send(json.dumps({"type": "hello", "settings": self.poll_settings()}))
            async for raw in websocket:
                message = json.loads(raw)
                if message.get("type") == "settings":
                    allowed = set(self._settings)
                    with self._lock:
                        self._settings.update({k: v for k, v in message.items() if k in allowed})
                    await self._broadcast({"type": "settings", "settings": self.poll_settings()})
        finally:
            self._clients.discard(websocket)

    async def _broadcast(self, payload):
        if not self._clients:
            return
        raw = json.dumps(payload, separators=(",", ":"))
        dead = []
        for client in tuple(self._clients):
            try:
                await client.send(raw)
            except Exception:
                dead.append(client)
        for client in dead:
            self._clients.discard(client)

    def poll_settings(self):
        with self._lock:
            return dict(self._settings)

    def has_clients(self):
        return bool(self._clients)

    def publish(self, iteration, gaussian_count, allocated_mib, reserved_mib,
                rgb_jpeg=None, geometry_jpeg=None):
        if self._loop is None or not self._clients:
            return
        payload = {
            "type": "frame",
            "iteration": int(iteration),
            "gaussian_count": int(gaussian_count),
            "vram_allocated_mib": round(float(allocated_mib), 2),
            "vram_reserved_mib": round(float(reserved_mib), 2),
        }
        if rgb_jpeg is not None:
            payload["rgb"] = base64.b64encode(rgb_jpeg).decode("ascii")
        if geometry_jpeg is not None:
            payload["geometry"] = base64.b64encode(geometry_jpeg).decode("ascii")
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)
