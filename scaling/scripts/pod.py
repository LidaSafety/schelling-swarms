#!/usr/bin/env python3
"""Start, check and stop the RunPod GPU pods that serve the swarm runs' models with vLLM: one pod per model, Qwen3-8B
by default, Gemma 3 27B with --model gemma-3-27b (a mixed swarm needs both at once), Qwen3-14B and Qwen3-32B
with --model qwen3-14b and --model qwen3-32b-bf16.

Keys come from a .env in the folder that contains this repository (RUNPOD_API or RUNPOD_API_KEY; OPENROUTER_API or
OPENROUTER_API_KEY),
written as NAME=value or NAME = value. Each pod's id, address and its own vLLM key go next to it, to .pod.env for
Qwen3-8B and to .pod-<model>.env for another model; scripts/run_on_pod.sh reads them. No command prints a key.

  python3 scripts/pod.py [--model M] gpus         prices and availability of GPUs that fit the model (read-only)
  python3 scripts/pod.py [--model M] up [GPU...]  create the pod: billing starts (POD_CLOUDS=SECURE: secure cloud only;
                                                  POD_GPU_COUNT=2: two GPUs, the model split across them)
  python3 scripts/pod.py [--model M] wait         poll until vLLM answers (image pulled, model downloaded and loaded)
  python3 scripts/pod.py [--model M] status
  python3 scripts/pod.py [--model M] down         terminate the pod: billing stops and nothing on it is kept
  python3 scripts/pod.py [--model M] use POD_ID   use a pod created in the console instead (key in .env as VLLM_API_KEY)
  eval "$(python3 scripts/pod.py endpoints)"      SB_ENDPOINTS for scb: address, key and request options of every pod
  eval "$(python3 scripts/pod.py exports)"              SB_BASE_URL / SB_API_KEY of one pod, for this shell
  eval "$(python3 scripts/pod.py exports openrouter)"   the same for OpenRouter
  eval "$(python3 scripts/pod.py exports anthropic)"    SB_BACKEND / SB_API_KEY for Claude (ANTHROPIC_API_KEY in .env)
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

def _beside_repo() -> Path:
    """The folder that contains this repository, wherever the script sits in it: keys and pod records live there, so
    they can never be committed."""
    here = Path(__file__).resolve()
    return next((p.parent for p in here.parents if (p / ".git").exists()), here.parents[2])


ROOT = _beside_repo()
ENV = ROOT / ".env"
# REST API v2 (v1 at rest.runpod.io was retired on 2026-11-15); GraphQL for GPU stock and container uptime
REST, GQL = "https://api.runpod.io/v2", "https://api.runpod.io/graphql"
IMAGE = "vllm/vllm-openai:v0.32.0-cu129"            # released 2026-10-18; the CUDA 12.9 build runs on more hosts
# 12.9+ natively; older drivers through CUDA compatibility, only the driver branches the image accepts
# (NVIDIA_REQUIRE_CUDA: 535, 550, 560, 565, 570 = CUDA 12.2, 12.4, 12.6, 12.7, 12.8). Datacenter GPUs only.
# Matched exactly by the API (a version no host reports gives "no instances available"); since November 2026 most
# secure-cloud hosts report 13.2, so leaving it out matched only older community hosts.
CUDA = ["13.2", "13.1", "13.0", "12.9", "12.8", "12.7", "12.6", "12.4", "12.2"]
# Datacenter GPUs only: a community RTX A6000 failed with "CUDA unknown error" at CUDA init (a host fault; workstation
# cards also lack CUDA forward compatibility). A100 first: the GPU the original results used.
GPUS = ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe",
        "NVIDIA H100 NVL", "NVIDIA L40S", "NVIDIA L40", "NVIDIA A40", "NVIDIA H200"]
BIG_GPUS = ["NVIDIA H200", "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe", "NVIDIA A100-SXM4-80GB",
            "NVIDIA A100 80GB PCIe"]                # 80 GB or more, fastest first: Gemma 3 27B's weights take 55 GB
# Served name -> Hugging Face repo, vLLM flags, request options sent with every call (scb's SB_ENDPOINTS, as in
# configs/models.yaml), GPUs that fit, and container disk in GB. Server flags as in scripts/sweep.py (default sampling
# settings), with room for crowded pages. google/gemma-3-27b-it is gated; unsloth/gemma-3-27b-it needs no Hugging Face
# token and has byte-identical weights (sha256 of all 12 shards), tokenizer.json, tokenizer.model and chat template,
# and the sampling defaults Google documents (top_k 64, top_p 0.95); checked 2026-11-08.
SERVERS = {
    "qwen3-8b": dict(repo="Qwen/Qwen3-8B", flags="--max-model-len 32768 --gpu-memory-utilization 0.90",
                     extra_body={"chat_template_kwargs": {"enable_thinking": False}}, gpus=GPUS, disk=80),
    "gemma-3-27b": dict(repo="unsloth/gemma-3-27b-it", flags="--max-model-len 16384 --gpu-memory-utilization 0.92",
                        extra_body={}, gpus=BIG_GPUS, disk=120),
    # Larger Qwen3 sizes, as the runs in ../wiki_game/data/swarm4round and the ruleoff tags: thinking off, Qwen3-32B
    # at full precision (served as qwen3-32b-bf16, 65 GB of weights)
    "qwen3-14b": dict(repo="Qwen/Qwen3-14B", flags="--max-model-len 16384 --gpu-memory-utilization 0.90",
                      extra_body={"chat_template_kwargs": {"enable_thinking": False}}, gpus=GPUS, disk=100),
    "qwen3-32b-bf16": dict(repo="Qwen/Qwen3-32B", flags="--max-model-len 16384 --gpu-memory-utilization 0.92",
                           extra_body={"chat_template_kwargs": {"enable_thinking": False}}, gpus=BIG_GPUS, disk=150),
}


# `--model both up`: one H200 (141 GB) serving both models of a mixed swarm, for when two GPUs are not free at once.
# Qwen3-8B starts first on port 8001 with a quarter of the memory, Gemma 3 27B on port 8000 once Qwen answers (two vLLM
# servers starting together would each count the other's memory as free). Each model gets its usual pod file.
BOTH_PORTS = {"gemma-3-27b": 8000, "qwen3-8b": 8001}
BOTH_START = (
    "rm -rf /usr/local/lib/python3.12/dist-packages/torchcodec*; "
    f"vllm serve {SERVERS['qwen3-8b']['repo']} --served-model-name qwen3-8b --port 8001 --max-model-len 32768 "
    "--gpu-memory-utilization 0.25 & "
    "for i in $(seq 180); do python3 -c \"import urllib.request as u; u.urlopen('http://localhost:8001/health')\" "
    "2>/dev/null && break; sleep 5; done; "
    f"exec vllm serve {SERVERS['gemma-3-27b']['repo']} --served-model-name gemma-3-27b --port 8000 "
    "--max-model-len 16384 --gpu-memory-utilization 0.62")


def pod_file(served: str) -> Path:
    """Where the pod serving `served` is recorded (.pod.env for Qwen3-8B, as before)."""
    return ROOT / (".pod.env" if served == "qwen3-8b" else f".pod-{served}.env")


def gpu_count() -> int:
    """GPUs per pod: POD_GPU_COUNT (default 1). With more than one, vLLM splits the model across them (tensor
    parallel), which also gives it the memory of all of them for long conversations."""
    v = os.environ.get("POD_GPU_COUNT", "1")
    if not v.isdigit() or not 1 <= int(v) <= 8:
        sys.exit("POD_GPU_COUNT takes a whole number from 1 to 8")
    return int(v)


def start_cmd(served: str) -> str:
    """The container's command. The image's CUDA 12.9 build ships a torchcodec whose library fails to load (OSError),
    which kills vLLM at import before it reaches the GPU. Text-only runs never use it, so it is removed first."""
    s, n = SERVERS[served], gpu_count()
    return (f"rm -rf /usr/local/lib/python3.12/dist-packages/torchcodec*; exec vllm serve {s['repo']} "
            f"--served-model-name {served} --port 8000 {s['flags']}" + (f" --tensor-parallel-size {n}" if n > 1 else ""))


def load_env(*paths: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*(.*?)\s*$", line)
            if m and not line.lstrip().startswith("#"):
                k, v = m.groups()
                out[k] = v[1:-1] if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"" else v
    return out


def key(env: dict[str, str], *names: str) -> str:
    for n in names:
        if env.get(n):
            return env[n]
    sys.exit(f"none of {', '.join(names)} is set in {ENV}")


def http(method: str, url: str, token: str | None, body: dict | None = None, timeout: float = 60):
    # RunPod's Cloudflare front rejects urllib's default user agent (error 1010)
    headers = {"Content-Type": "application/json", "User-Agent": "swarm-pod-script/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt.strip() else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:1500]
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return 0, str(e)


def pod_env(served: str) -> dict[str, str]:
    env = load_env(pod_file(served))
    if not env.get("POD_ID"):
        sys.exit(f"no pod recorded in {pod_file(served)}: run `up` first")
    return env


def cmd_gpus(rp: str, served: str) -> None:
    q = ("query { gpuTypes { id displayName memoryInGb securePrice communityPrice "
         "lowestPrice(input: {gpuCount: 1}) { stockStatus uninterruptablePrice } } }")
    code, d = http("POST", GQL, rp, {"query": q})
    if code != 200 or not isinstance(d, dict) or "data" not in d:
        sys.exit(f"GPU query failed: HTTP {code} {str(d)[:300]}")
    by_id = {g["id"]: g for g in d["data"]["gpuTypes"]}
    print(f"{'GPU':26s} {'GB':>4s} {'secure $/h':>11s} {'community $/h':>14s}  stock")
    for gid in SERVERS[served]["gpus"]:
        g = by_id.get(gid)
        if g:
            lp = g.get("lowestPrice") or {}
            print(f"{gid:26s} {g['memoryInGb']:>4} {g.get('securePrice') or 0:>11.2f} "
                  f"{g.get('communityPrice') or 0:>14.2f}  {lp.get('stockStatus')}")


def cmd_up(rp: str, gpus: list[str], served: str) -> None:
    names = list(BOTH_PORTS) if served == "both" else [served]
    for n in names:
        if pod_file(n).exists() and load_env(pod_file(n)).get("POD_ID"):
            sys.exit(f"a pod is already recorded in {pod_file(n)}; run `down` first")
    vkey = secrets.token_urlsafe(24)
    gpus = gpus or (["NVIDIA H200"] if served == "both" else SERVERS[served]["gpus"])
    # v2 takes one GPU type per pod, so try the types in order ourselves (v1 did this with gpuTypePriority "custom")
    # community: cheaper, more often free, less reliable; POD_CLOUDS=SECURE asks for the secure cloud only
    clouds = os.environ.get("POD_CLOUDS", "SECURE COMMUNITY").split()
    if not clouds or any(c not in ("SECURE", "COMMUNITY") for c in clouds):
        sys.exit("POD_CLOUDS takes SECURE and/or COMMUNITY")
    tries = [(c, g) for c in clouds for g in gpus]
    for cloud, gpu in tries:
        code, d = _create(rp, cloud, gpu, vkey, served)
        if code in (200, 201) and isinstance(d, dict) and d.get("id"):
            break
        print(f"{cloud} {gpu}: HTTP {code} {str(d)[:160]}", flush=True)
    else:
        sys.exit("no instance available in either cloud; try again in a few minutes or add GPU types")
    pid = d["id"]
    for n in names:
        port = BOTH_PORTS[n] if served == "both" else 8000
        pod_file(n).write_text(f"POD_ID={pid}\nSB_BASE_URL=https://{pid}-{port}.proxy.runpod.net/v1\nSB_API_KEY={vkey}\n")
        pod_file(n).chmod(0o600)
    print(f"pod {pid} ({' + '.join(names)}) created in {cloud} cloud: {(d.get('gpu') or {}).get('id', gpu)}, "
          f"${d.get('cost')}/h; address and key saved to {', '.join(str(pod_file(n)) for n in names)}")


def _create(rp: str, cloud: str, gpu: str, vkey: str, served: str):
    both = served == "both"
    body = {   # REST v2 (strict: unknown fields are refused); on-demand, no volume
        "name": "swarm-" + ("-".join(BOTH_PORTS) if both else served), "image": IMAGE, "cloud": cloud,
        "gpu": {"id": gpu, "count": 1 if both else gpu_count(), "allowedCudaVersions": CUDA},
        "disk": 150 if both else SERVERS[served]["disk"],
        "ports": [f"{p}/http" for p in BOTH_PORTS.values()] if both else ["8000/http"],
        "env": {"VLLM_API_KEY": vkey},     # vLLM's --api-key default: the public proxy URL needs a key
        "entrypoint": ["bash", "-c"],
        "cmd": [BOTH_START if both else start_cmd(served)],
    }
    return http("POST", f"{REST}/pods", rp, body)


def runtime(rp: str, pid: str) -> tuple[int | None, int | None]:
    """(container uptime in s, GPU memory in use %) from RunPod; (None, None) if unknown."""
    q = 'query { pod(input: {podId: "%s"}) { runtime { uptimeInSeconds gpus { memoryUtilPercent } } } }' % pid
    _, d = http("POST", GQL, rp, {"query": q})
    try:
        rt = d["data"]["pod"]["runtime"]
        return rt["uptimeInSeconds"], (rt.get("gpus") or [{}])[0].get("memoryUtilPercent")
    except (TypeError, KeyError, IndexError):
        return None, None


def cmd_wait(rp: str, served: str, timeout_min: float = 25, start_min: float = 12) -> None:
    """Poll until vLLM answers. Every 30 s also read the container's uptime: if it falls back twice, the
    container is crash-looping, and if the container has not started after `start_min` minutes the host is stuck
    (a community A100 once never pulled the image in 40 minutes); waiting longer will not help either way."""
    pe = pod_env(served)
    base, vkey, t0 = pe["SB_BASE_URL"], pe["SB_API_KEY"], time.time()
    last, prev_up, restarts, polls = None, None, 0, 0
    while time.time() - t0 < timeout_min * 60:
        polls += 1
        if polls % 2 == 0:
            up, mem = runtime(rp, pe["POD_ID"])
            if up is not None and prev_up is not None and up < prev_up:
                restarts += 1
            prev_up = up if up is not None else prev_up
            print(f"{time.time() - t0:5.0f}s  container up {up}s, GPU memory {mem}%, restarts seen {restarts}",
                  flush=True)
            if restarts >= 2:       # vLLM never restarts on its own while loading: a host fault or a bad start
                sys.exit("the container keeps restarting: read the pod's logs in the RunPod console")
            if prev_up is None and time.time() - t0 > start_min * 60:
                sys.exit(f"the container has not started after {start_min:.0f} min: the host looks stuck; "
                         f"run `down` and try another GPU")
        code, d = http("GET", f"{base}/models", vkey, timeout=20)
        if code == 200 and isinstance(d, dict) and any(m.get("id") == served for m in d.get("data", [])):
            anon, _ = http("GET", f"{base}/models", None, timeout=20)
            _, ver = http("GET", base.removesuffix("/v1") + "/version", vkey, timeout=20)
            print(f"ready after {time.time() - t0:.0f}s: model {served}, vLLM {ver.get('version') if isinstance(ver, dict) else ver}; "
                  f"request without the key -> HTTP {anon} ({'protected' if anon == 401 else 'NOT protected'})",
                  flush=True)
            return
        if code != last:
            print(f"{time.time() - t0:5.0f}s  not ready (HTTP {code})", flush=True)
            last = code
        time.sleep(15)
    sys.exit(f"not ready after {timeout_min} min; check the pod's logs in the RunPod console")


def cmd_status(rp: str, served: str) -> None:
    pid = pod_env(served)["POD_ID"]
    code, d = http("GET", f"{REST}/pods/{pid}", rp)
    if code != 200 or not isinstance(d, dict):
        sys.exit(f"HTTP {code} {str(d)[:300]}")
    print(f"pod {pid}: {d.get('status')}, {(d.get('gpu') or {}).get('id')}, ${d.get('cost')}/h, "
          f"created {d.get('createdAt')}, started {d.get('startedAt')}")


def cmd_use(pid: str, served: str) -> None:
    """Point the runs at a pod created elsewhere (e.g. in the RunPod console), serving vLLM on HTTP port 8000 with the
    key in .env as VLLM_API_KEY. `down` will not terminate such a pod."""
    pf = pod_file(served)
    if pf.exists() and load_env(pf).get("POD_ID"):
        sys.exit(f"a pod is already recorded in {pf}; run `down` (or delete the file) first")
    vkey = key(load_env(ENV), "VLLM_API_KEY")
    pf.write_text(f"POD_ID={pid}\nSB_BASE_URL=https://{pid}-8000.proxy.runpod.net/v1\nSB_API_KEY={vkey}\nOWNER=user\n")
    pf.chmod(0o600)
    print(f"runs will use pod {pid} for {served}; check it with `wait`")


def cmd_down(rp: str, served: str) -> None:
    pe = pod_env(served)
    pid = pe["POD_ID"]
    files = [pod_file(n) for n in SERVERS if load_env(pod_file(n)).get("POD_ID") == pid]   # a pod may serve two models
    if pe.get("OWNER") == "user":
        for f in files:
            f.unlink()
        sys.exit(f"pod {pid} was created by you, so I leave it running; stop it in the RunPod console")
    code, d = http("DELETE", f"{REST}/pods/{pid}", rp)
    if code not in (200, 204, 404):
        sys.exit(f"terminate failed: HTTP {code} {str(d)[:300]}; terminate pod {pid} in the RunPod console")
    for f in files:
        f.unlink()
    print(f"pod {pid} ({served}) " + ("was already gone" if code == 404 else "terminated; billing stopped"))


def cmd_exports(target: str, served: str) -> None:
    if target == "openrouter":
        k = key(load_env(ENV), "OPENROUTER_API_KEY", "OPENROUTER_API")
        pairs = {"SB_BASE_URL": "https://openrouter.ai/api/v1", "SB_API_KEY": k}
    elif target == "anthropic":   # Claude through Anthropic's Messages API (scb/llm.py SB_BACKEND)
        pairs = {"SB_BACKEND": "anthropic", "SB_API_KEY": key(load_env(ENV), "ANTHROPIC_API_KEY", "ANTHROPIC_API")}
    else:
        pe = pod_env(served)
        pairs = {"SB_BASE_URL": pe["SB_BASE_URL"], "SB_API_KEY": pe["SB_API_KEY"]}
    print("; ".join(f"export {k}={shlex.quote(v)}" for k, v in pairs.items()))


def cmd_endpoints() -> None:
    """SB_ENDPOINTS for scb (scb/cli.py): the address, key and request options of every pod recorded here, per model."""
    eps = {}
    for served, s in SERVERS.items():
        pe = load_env(pod_file(served))
        if pe.get("POD_ID"):
            eps[served] = {"base_url": pe["SB_BASE_URL"], "api_key": pe["SB_API_KEY"], "extra_body": s["extra_body"]}
    if not eps:
        sys.exit("no pod recorded: run `up` first")
    print(f"export SB_ENDPOINTS={shlex.quote(json.dumps(eps))}")


def main() -> None:
    args, served = sys.argv[1:], "qwen3-8b"
    if "--model" in args:
        i = args.index("--model")
        served = args[i + 1] if i + 1 < len(args) else ""
        del args[i:i + 2]
    args = args or ["status"]
    cmd, rest = args[0], args[1:]
    if served not in SERVERS and not (served == "both" and cmd == "up"):
        sys.exit(f"--model must be one of: {', '.join(SERVERS)} (or both, with up)")
    if cmd == "exports":
        return cmd_exports(rest[0] if rest else "pod", served)
    if cmd == "endpoints":
        return cmd_endpoints()
    if cmd == "use":
        return cmd_use(rest[0], served) if rest else sys.exit("usage: pod.py [--model M] use <pod-id>")
    rp = key(load_env(ENV), "RUNPOD_API_KEY", "RUNPOD_API")
    if cmd == "wait":
        # POD_START_MIN: minutes to wait for the container to start (default 12); a host without the image cached
        # can take longer than that just to unpack it
        return cmd_wait(rp, served, float(rest[0]) if rest else 25, float(os.environ.get("POD_START_MIN", "12")))
    {"gpus": lambda: cmd_gpus(rp, served), "up": lambda: cmd_up(rp, rest, served),
     "status": lambda: cmd_status(rp, served), "down": lambda: cmd_down(rp, served)}.get(cmd, lambda: sys.exit(__doc__))()


if __name__ == "__main__":
    main()
