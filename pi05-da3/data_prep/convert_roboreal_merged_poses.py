#!/usr/bin/env python3
"""Fast PARALLEL converter: RoboReal HDF5 -> a single MERGED LeRobot v2.0 dataset.

Why this exists (vs scripts/convert_roboreal_to_lerobot.py + convert_all_roboreal.sh):
  * The "official" converter is serial per episode and only parallelises at the
    (scene,task) level via a 4-wide bash pool. On 16k episodes that is slow.
  * This version fans every episode out across a ProcessPoolExecutor (default
    one worker per ~4 cores), so all 240 cores can be put to work, and produces
    ONE merged dataset (global episode_index / frame index / chunking) instead
    of hundreds of per-task datasets.

Differences requested for this run:
  * Cameras kept: left, right, countertop  (head/front/demo dropped).
  * Output video keys use LITERAL names: observation.images.{left,right,countertop}
    (not the cam_left_wrist/... convention).
  * Converts ALL 16k trajectories = every <scene>/<task>/<variant>/data/episode*.hdf5
    where variant in {clean, d6..d15}.

RoboReal HDF5 schema (verified):
    /joint_action/vector                  (T, 14) float64  absolute joint targets
    /observation/countertop_camera/rgb    (T,) bytes (JPEG)
    /observation/left_camera/rgb          (T,) bytes (JPEG)
    /observation/right_camera/rgb         (T,) bytes (JPEG)
  (no qpos -> observation.state is a one-step-delayed copy of action, matching
   the official converter's convention.)
Per-episode instruction: <variant>/instructions/episode<N>.json -> seen[0].

Output (LeRobot v2.0, multi-chunk, 1000 episodes/chunk):
  <out>/meta/{info.json,episodes.jsonl,tasks.jsonl,modality.json}
  <out>/data/chunk-{c:03d}/episode_{g:06d}.parquet
  <out>/videos/chunk-{c:03d}/observation.images.{cam}/episode_{g:06d}.mp4

Idempotent / resumable: an episode whose 3 mp4s + parquet already exist and are
non-empty is skipped, so the job can be re-run after an interruption.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # fall back to PATH
    FFMPEG = "ffmpeg"

# RoboReal camera name -> LITERAL LeRobot video key suffix
CAMERA_KEY_MAP = {
    "countertop_camera": "countertop",
    "left_camera":       "left",
    "right_camera":      "right",
}
FPS = 25
ACTION_DIM = 14
CHUNK_SIZE = 1000

cv2.setNumThreads(1)  # avoid thread oversubscription inside pool workers


# --------------------------------------------------------------------------- #
# Episode enumeration
# --------------------------------------------------------------------------- #
def variant_rank(variant: str) -> int:
    """clean first (-1), then d6,d7,... numerically."""
    if variant == "clean":
        return -1
    if variant.startswith("d") and variant[1:].isdigit():
        return int(variant[1:])
    return 10_000  # unknown variants last, stable


def enumerate_episodes(root: Path):
    """Return list of dicts in a DETERMINISTIC global order.

    Each: {scene, task, variant, ep_num, hdf5, instr}
    """
    eps = []
    for scene in sorted(os.listdir(root)):
        sp = root / scene
        if not sp.is_dir():
            continue
        for task in sorted(os.listdir(sp)):
            tp = sp / task
            if not tp.is_dir():
                continue
            for variant in sorted(os.listdir(tp)):
                data_dir = tp / variant / "data"
                if not data_dir.is_dir():
                    continue
                instr_dir = tp / variant / "instructions"
                for f in os.listdir(data_dir):
                    if not (f.startswith("episode") and f.endswith(".hdf5")):
                        continue
                    ep_num = int(f[len("episode"):-len(".hdf5")])
                    eps.append({
                        "scene": scene, "task": task, "variant": variant,
                        "ep_num": ep_num,
                        "hdf5": str(data_dir / f),
                        "instr": str(instr_dir / f"episode{ep_num}.json"),
                    })
    eps.sort(key=lambda e: (e["scene"], e["task"], variant_rank(e["variant"]), e["ep_num"]))
    return eps


def pick_instruction(instr_path: str, fallback: str, rng: random.Random) -> str:
    p = Path(instr_path)
    if not p.exists():
        return fallback
    try:
        payload = json.loads(p.read_text())
    except Exception:
        return fallback
    seen = payload.get("seen") or payload.get("instructions") or []
    if not seen:
        return fallback
    return seen[0] if len(seen) == 1 else rng.choice(seen)


# --------------------------------------------------------------------------- #
# Pass 1 worker: cheap metadata (episode length T) — no image decode
# --------------------------------------------------------------------------- #
def read_length(hdf5_path: str):
    """Validate an episode and return (T, None) on success or (-1, err) on failure.

    Validation = openable + has /joint_action/vector with dim-14 + all 3 cameras.
    A corrupt/zero-byte file or one missing a camera is reported, not raised, so
    the whole 16k job is never killed by a single bad episode.
    """
    try:
        with h5py.File(hdf5_path, "r") as f:
            shp = f["/joint_action/vector"].shape
            if len(shp) != 2 or shp[1] != ACTION_DIM:
                return -1, f"bad action shape {shp}"
            for src in CAMERA_KEY_MAP:
                if f"/observation/{src}/rgb" not in f:
                    return -1, f"missing /observation/{src}/rgb"
            return int(shp[0]), None
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# Pass 2 worker: decode cameras, encode mp4s, write parquet
# --------------------------------------------------------------------------- #
def decode_rgb_frames(byte_dataset) -> np.ndarray:
    frames = []
    for raw in byte_dataset:
        arr = np.frombuffer(raw, dtype=np.uint8)
        # NOTE: the source JPEGs are stored channel-swapped (non-standard capture), so cv2.imdecode's
        # output is ALREADY display-correct when treated as RGB. Do NOT cvtColor(BGR2RGB) here — that
        # produced BGR-swapped videos (verified: beige cabinets rendered blue). Feed straight to the
        # rgb24 ffmpeg pipe. (The pre-2026-07 RoboReal LeRobot datasets were built WITH the bad swap;
        # load them with DA3CacheConfig(bgr_to_rgb=True) to correct at train time.)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("cv2.imdecode returned None")
        frames.append(img)
    return np.stack(frames, axis=0)


def _ffmpeg_proc(out_path: Path, H: int, W: int, fps: int, crf: int, preset: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", preset,
        "-crf", str(crf), "-threads", "1", str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def outputs_exist(out_dir: Path, chunk: str, gidx: int) -> bool:
    parq = out_dir / "data" / chunk / f"episode_{gidx:06d}.parquet"
    if not (parq.exists() and parq.stat().st_size > 0):
        return False
    for cam in CAMERA_KEY_MAP.values():
        mp4 = out_dir / "videos" / chunk / f"observation.images.{cam}" / f"episode_{gidx:06d}.mp4"
        if not (mp4.exists() and mp4.stat().st_size > 0):
            return False
    return True


def convert_episode(job: dict) -> dict:
    """job: gidx, hdf5, frame_offset, task_index, instruction, out_dir, crf, preset"""
    gidx = job["gidx"]
    out_dir = Path(job["out_dir"])
    chunk = f"chunk-{gidx // CHUNK_SIZE:03d}"

    if job.get("resume") and outputs_exist(out_dir, chunk, gidx):
        return {"gidx": gidx, "length": job["T"], "instruction": job["instruction"], "skipped": True}

    cv2.setNumThreads(1)
    with h5py.File(job["hdf5"], "r") as f:
        action = np.asarray(f["/joint_action/vector"][:], dtype=np.float32)
        T = action.shape[0]
        assert action.shape == (T, ACTION_DIM), f"bad action {action.shape} in {job['hdf5']}"
        # one-step-delayed action as state (no qpos in roboreal)
        state = np.empty_like(action)
        state[0] = action[0]
        state[1:] = action[:-1]
        cam_frames = {}
        for src, dst in CAMERA_KEY_MAP.items():
            key = f"/observation/{src}/rgb"
            if key not in f:
                raise KeyError(f"{job['hdf5']}: missing {key}")
            cam_frames[dst] = decode_rgb_frames(f[key])
        cam_extr = {}; cam_intr = {}
        for src, dst in CAMERA_KEY_MAP.items():
            ek = f"/observation/{src}/extrinsic_cv"; ik = f"/observation/{src}/intrinsic_cv"
            if ek not in f or ik not in f:
                raise KeyError(f"{job['hdf5']}: missing {ek} or {ik}")
            ex = np.asarray(f[ek][:], dtype=np.float32)   # (Te,3,4)
            ic = np.asarray(f[ik][:], dtype=np.float32)   # (Ti,3,3)
            # Check-1: pose T must cover action T; slice to T (never pad)
            if ex.shape[0] < T or ic.shape[0] < T:
                raise ValueError(f"{job['hdf5']}: pose T {ex.shape[0]}/{ic.shape[0]} < action T {T}")
            cam_extr[dst] = ex[:T].reshape(T, 12)
            cam_intr[dst] = ic[:T].reshape(T, 9)

    # Encode all 3 cameras concurrently (overlap subprocess pipes)
    procs = []
    for dst, frames in cam_frames.items():
        Tc, H, W, _ = frames.shape
        mp4 = out_dir / "videos" / chunk / f"observation.images.{dst}" / f"episode_{gidx:06d}.mp4"
        p = _ffmpeg_proc(mp4, H, W, FPS, job["crf"], job["preset"])
        procs.append((dst, p, frames, mp4))
    for dst, p, frames, mp4 in procs:
        p.stdin.write(frames.tobytes())
        p.stdin.close()
    for dst, p, frames, mp4 in procs:
        if p.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {mp4}")

    # Parquet (global index from precomputed frame_offset)
    off = job["frame_offset"]
    tbl = pa.table({
        "action":            pa.array([a.tolist() for a in action], type=pa.list_(pa.float32(), ACTION_DIM)),
        "observation.state": pa.array([s.tolist() for s in state],  type=pa.list_(pa.float32(), ACTION_DIM)),
        "task_index":        pa.array(np.full(T, job["task_index"], dtype=np.int64)),
        "episode_index":     pa.array(np.full(T, gidx, dtype=np.int64)),
        "frame_index":       pa.array(np.arange(T, dtype=np.int64)),
        "timestamp":         pa.array(np.arange(T, dtype=np.float64) / FPS),
        "index":             pa.array(np.arange(off, off + T, dtype=np.int64)),
        "observation.left.extrinsic_cv": pa.array([r.tolist() for r in cam_extr["left"]], type=pa.list_(pa.float32(), 12)),
        "observation.left.intrinsic_cv": pa.array([r.tolist() for r in cam_intr["left"]], type=pa.list_(pa.float32(), 9)),
        "observation.right.extrinsic_cv": pa.array([r.tolist() for r in cam_extr["right"]], type=pa.list_(pa.float32(), 12)),
        "observation.right.intrinsic_cv": pa.array([r.tolist() for r in cam_intr["right"]], type=pa.list_(pa.float32(), 9)),
        "observation.countertop.extrinsic_cv": pa.array([r.tolist() for r in cam_extr["countertop"]], type=pa.list_(pa.float32(), 12)),
        "observation.countertop.intrinsic_cv": pa.array([r.tolist() for r in cam_intr["countertop"]], type=pa.list_(pa.float32(), 9)),
    })
    parq = out_dir / "data" / chunk / f"episode_{gidx:06d}.parquet"
    parq.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, parq, compression="snappy")
    return {"gidx": gidx, "length": T, "instruction": job["instruction"], "skipped": False}


# --------------------------------------------------------------------------- #
# Meta writers
# --------------------------------------------------------------------------- #
def write_modality(out_dir: Path):
    base = {
        "action": {
            "left_joints":  {"start": 0,  "end": 6,  "original_key": "action"},
            "left_gripper": {"start": 6,  "end": 7,  "original_key": "action"},
            "right_joints": {"start": 7,  "end": 13, "original_key": "action"},
            "right_gripper":{"start": 13, "end": 14, "original_key": "action"},
        },
        "state": {
            "left_joints":  {"start": 0,  "end": 6,  "original_key": "observation.state"},
            "left_gripper": {"start": 6,  "end": 7,  "original_key": "observation.state"},
            "right_joints": {"start": 7,  "end": 13, "original_key": "observation.state"},
            "right_gripper":{"start": 13, "end": 14, "original_key": "observation.state"},
        },
        "video": {
            cam: {"original_key": f"observation.images.{cam}"}
            for cam in CAMERA_KEY_MAP.values()
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
    (out_dir / "meta" / "modality.json").write_text(json.dumps(base, indent=4))


def write_meta(out_dir: Path, n_eps: int, total_frames: int, tasks: dict,
               episodes_meta: list, H: int, W: int):
    info = {
        "codebase_version": "v2.0",
        "robot_type": "roboreal",
        "total_episodes": n_eps,
        "total_frames": int(total_frames),
        "total_tasks": len(tasks),
        "total_videos": n_eps * len(CAMERA_KEY_MAP),
        "total_chunks": (n_eps + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": None},
            "observation.state": {"dtype": "float32", "shape": [ACTION_DIM], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    for cam in CAMERA_KEY_MAP.values():
        info["features"][f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [3, H, W],
            "names": ["channels", "height", "width"],
            "video_info": {
                "video.fps": float(FPS), "video.codec": "h264",
                "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    for cam in CAMERA_KEY_MAP.values():
        info["features"][f"observation.{cam}.extrinsic_cv"] = {"dtype": "float32", "shape": [12], "names": None}
        info["features"][f"observation.{cam}.intrinsic_cv"] = {"dtype": "float32", "shape": [9], "names": None}
    (out_dir / "meta" / "camera_conventions.md").write_text(
        "# Camera conventions\n\n"
        "extrinsic_cv: OpenCV world->camera, flattened 3x4 (row-major), per-frame.\n"
        "intrinsic_cv: OpenCV K, flattened 3x3 (row-major), per-frame, calibrated to the stored native video resolution (no resize applied).\n"
        "cam_high == countertop. cam_left_wrist == left (MOVING). cam_right_wrist == right (MOVING).\n"
        "OpenGL cam2world_gl is intentionally NOT written.\n")
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    with open(out_dir / "meta" / "episodes.jsonl", "w") as fp:
        for e in episodes_meta:
            fp.write(json.dumps(e) + "\n")
    with open(out_dir / "meta" / "tasks.jsonl", "w") as fp:
        for text, tidx in sorted(tasks.items(), key=lambda kv: kv[1]):
            fp.write(json.dumps({"task_index": tidx, "task": text}) + "\n")
    write_modality(out_dir)


def probe_hw(out_dir: Path) -> tuple[int, int]:
    first_cam = next(iter(CAMERA_KEY_MAP.values()))
    for chunk_dir in sorted((out_dir / "videos").glob("chunk-*")):
        for mp4 in sorted((chunk_dir / f"observation.images.{first_cam}").glob("*.mp4")):
            cap = cv2.VideoCapture(str(mp4))
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if h > 0 and w > 0:
                    return h, w
            cap.release()
    return 480, 640


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=Path("/work/jack/roboreal_data"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=min(60, os.cpu_count() or 8))
    ap.add_argument("--limit", type=int, default=None, help="cap #episodes (smoke test)")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip episodes whose mp4s+parquet already exist")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)

    print(f"[enum] scanning {args.raw_root} ...", flush=True)
    eps = enumerate_episodes(args.raw_root)
    if args.limit:
        eps = eps[: args.limit]
    n = len(eps)
    print(f"[enum] {n} episodes (ffmpeg={FFMPEG})", flush=True)
    if n == 0:
        sys.exit("no episodes found")

    # ---- Pass 1: validate + episode lengths (parallel, cheap) ----
    print(f"[pass1] validating + reading lengths with {args.workers} workers ...", flush=True)
    lengths = [-1] * n
    errors = [None] * n
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(read_length, e["hdf5"]): i for i, e in enumerate(eps)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            lengths[i], errors[i] = fut.result()
            done += 1
            if done % 1000 == 0:
                print(f"[pass1] {done}/{n}", flush=True)

    # Drop unreadable/invalid episodes; keep only good ones in deterministic order.
    good = [i for i in range(n) if lengths[i] > 0]
    dropped = [i for i in range(n) if lengths[i] <= 0]
    if dropped:
        dpath = out_dir / "meta" / "dropped_episodes.txt"
        with open(dpath, "w") as fp:
            for i in dropped:
                fp.write(f"{eps[i]['hdf5']}\t{errors[i]}\n")
        print(f"[pass1] DROPPED {len(dropped)} bad episodes -> {dpath}", flush=True)
        for i in dropped:
            print(f"  DROP {eps[i]['hdf5']} :: {errors[i]}", flush=True)
    eps = [eps[i] for i in good]
    lengths = [lengths[i] for i in good]
    n = len(eps)

    # global frame offsets + task index map (deterministic, global order)
    rng = random.Random(args.seed)
    tasks: dict[str, int] = {}
    offsets = [0] * n
    running = 0
    jobs = []
    for i, e in enumerate(eps):
        offsets[i] = running
        running += lengths[i]
        instr = pick_instruction(e["instr"], f"{e['scene']}_{e['task']}", rng)
        if instr not in tasks:
            tasks[instr] = len(tasks)
        jobs.append({
            "gidx": i, "hdf5": e["hdf5"], "frame_offset": offsets[i],
            "task_index": tasks[instr], "instruction": instr, "T": lengths[i],
            "out_dir": str(out_dir), "crf": args.crf, "preset": args.preset,
            "resume": args.resume,
        })
    total_frames = running
    print(f"[plan] {n} eps, {total_frames} frames, {len(tasks)} unique tasks", flush=True)

    # ---- Pass 2: decode + encode + parquet (parallel, heavy) ----
    episodes_meta = [None] * n
    done = skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(convert_episode, j): j["gidx"] for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            episodes_meta[r["gidx"]] = {
                "episode_index": r["gidx"], "tasks": [r["instruction"]], "length": r["length"],
            }
            done += 1
            skipped += int(r.get("skipped", False))
            if done % 200 == 0 or done == n:
                print(f"[pass2] {done}/{n} (skipped {skipped})", flush=True)

    H, W = probe_hw(out_dir)
    write_meta(out_dir, n, total_frames, tasks, episodes_meta, H, W)
    print(f"[DONE] {n} eps, {total_frames} frames, {len(tasks)} tasks, {H}x{W} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
