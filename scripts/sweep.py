"""Serve each model with vLLM, run its experiments, stop the server.

  python scripts/sweep.py                                   # every model in configs/models.yaml, every experiment
  python scripts/sweep.py --models qwen3-8b --only deaddrop_drive committee_sealed
  python scripts/sweep.py --models qwen3-8b --external http://host:8000/v1   # a server that is already running
  python scripts/sweep.py --dry-run                         # print the commands only

An experiment is skipped if <out>/.done/<model>__<experiment> exists, so a sweep can be resumed. Logs go to
<out>/logs/. Needs `pip install vllm` on a GPU machine (not needed with --external).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_vllm(m: dict, port: int, log: Path) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", m["hf_id"],
           "--served-model-name", m["name"], "--port", str(port),
           "--tensor-parallel-size", str(m.get("tp", 1)),
           "--max-model-len", str(m.get("max_len", 16384)),
           "--gpu-memory-utilization", str(m.get("gpu_mem", 0.90)),
           *m.get("vllm_args", [])]
    log.parent.mkdir(parents=True, exist_ok=True)
    print("  $", " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True)


def wait_ready(base: str, proc: subprocess.Popen | None, log: Path, timeout: float = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-15:]) if log.exists() else ""
            raise RuntimeError(f"vLLM exited with code {proc.returncode} before becoming ready:\n{tail}")
        try:
            with urllib.request.urlopen(base + "/models", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(10)
    raise TimeoutError(f"vLLM server at {base} not ready after {timeout}s")


def stop(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=120)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


def applies(e: dict, model: str) -> bool:
    return (not e.get("only_models") or model in e["only_models"]) and model not in e.get("skip_models", [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-file", default=str(ROOT / "configs" / "models.yaml"))
    ap.add_argument("--experiments-file", default=str(ROOT / "configs" / "experiments.yaml"))
    ap.add_argument("--models", nargs="*", help="subset of model names")
    ap.add_argument("--only", nargs="*", help="subset of experiment names")
    ap.add_argument("--external", help="use an already-running OpenAI-compatible server at this base URL")
    ap.add_argument("--api-key", default=os.environ.get("SB_API_KEY", "EMPTY"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    models = [m for m in yaml.safe_load(open(a.models_file))["models"] if not a.models or m["name"] in a.models]
    cfg = yaml.safe_load(open(a.experiments_file))
    out = (ROOT / cfg.get("out", "results")).resolve()
    exps = [e for e in cfg["experiments"] if not a.only or e["name"] in a.only]
    for m in models:
        todo = [e for e in exps if applies(e, m["name"]) and not (out / ".done" / f"{m['name']}__{e['name']}").exists()]
        print(f"\n=== {m['name']}: {len(todo)} experiments to run", flush=True)
        if not todo:
            continue
        port = free_port()
        base = a.external or f"http://127.0.0.1:{port}/v1"
        vlog = out / "logs" / f"vllm_{m['name']}.log"
        server = None
        try:
            if not a.external and not a.dry_run:
                server = start_vllm(m, port, vlog)
                wait_ready(base, server, vlog)
                print("  server ready", flush=True)
            env = {**os.environ, "SB_BASE_URL": base, "SB_API_KEY": a.api_key,
                   "SB_EXTRA_BODY": json.dumps(m.get("extra_body", {})), "SB_TOKEN_MULT": str(m.get("token_mult", 1))}
            for e in todo:
                cmd = [sys.executable, "-m", "scb", e["cmd"], "--models", m["name"],
                       "--workers", str(m.get("workers", 16)), "--out", str(out), *shlex.split(e["args"])]
                print(f"  [{m['name']} | {e['name']}] $ {' '.join(cmd[2:])}", flush=True)
                if a.dry_run:
                    continue
                log = out / "logs" / f"{m['name']}__{e['name']}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                with log.open("w") as f:
                    rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
                if rc == 0:
                    (out / ".done").mkdir(parents=True, exist_ok=True)
                    (out / ".done" / f"{m['name']}__{e['name']}").write_text(f"{time.time() - t0:.0f}s\n")
                    print(f"    ok ({time.time() - t0:.0f}s)", flush=True)
                else:
                    print(f"    FAILED (exit {rc}), see {log}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIPPING {m['name']}: {e}", flush=True)
        finally:
            if server is not None:
                stop(server)


if __name__ == "__main__":
    main()
