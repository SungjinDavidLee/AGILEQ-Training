import os
import signal
import subprocess
import time
from pathlib import Path


def launch_server(root, port, fps, log_path):
    root = Path(root).expanduser().resolve()
    executable = root / "CarlaUE4.sh"
    if not executable.is_file():
        raise FileNotFoundError(f"CARLA launcher not found: {executable}; use --carla-root or --connect")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as stream:
        return subprocess.Popen([
            str(executable), "-quality_level=Low", "-RenderOffScreen", "-benchmark",
            f"-fps={fps:g}", f"-carla-rpc-port={port}",
        ], cwd=root, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)


def wait_for_server(client, process, timeout):
    deadline = time.monotonic() + timeout
    client.set_timeout(2.0)
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError("CARLA exited during startup; check carla_server.log")
        try:
            client.get_world()
            return
        except RuntimeError:
            time.sleep(0.5)
    raise TimeoutError(f"CARLA did not become ready within {timeout}s")


def stop_server(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
