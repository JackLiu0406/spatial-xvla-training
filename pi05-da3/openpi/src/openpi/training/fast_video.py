"""Fast drop-in replacement for LeRobot's decode_video_frames.

LeRobot's `pyav` path builds a fresh `torchvision.io.VideoReader` on every call (per camera,
per sample) — instantiation alone is ~250 ms, which dominates data loading (~800 ms/sample here)
and starves the GPUs. This uses `av.open` directly (~15 ms) with the identical frame-selection
logic (seek to keyframe before first ts, decode forward, match each query ts to the closest
loaded frame within tolerance), so the output is bit-identical but ~15x faster.
"""

import time

import av
import torch


def _decode_once(video_path, timestamps):
    """Single attempt: open -> seek -> decode the needed frames -> close. Raises on failure."""
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        # SINGLE-THREADED decode on purpose: the videos are re-encoded GOP=1 (every frame a
        # keyframe), so each call decodes exactly one frame — multithreaded decode buys nothing
        # and spawns many ffmpeg/swscaler threads. Across 8 workers x N ranks that thread pileup
        # exhausts host resources and makes the swscaler fail with EAGAIN ("Resource temporarily
        # unavailable"), which — uncaught — kills a rank and cascades the whole jax.distributed run.
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        tb = stream.time_base

        first_ts = min(timestamps)
        last_ts = max(timestamps)

        # Seek to the keyframe at/before the first requested timestamp, then decode forward.
        container.seek(int(first_ts / tb), backward=True, stream=stream)

        loaded_frames = []
        loaded_ts = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_ts = float(frame.pts * tb)
            loaded_frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break
    finally:
        container.close()
    return loaded_frames, loaded_ts


def decode_video_frames_av(video_path, timestamps, tolerance_s, backend=None, log_loaded_timestamps=False):
    video_path = str(video_path)

    # Ride out host-resource failures (EAGAIN from the ffmpeg swscaler when the shared node is
    # thread/memory-starved by co-located jobs) instead of letting them crash the process and
    # cascade-kill the whole jax.distributed run. On this node the starvation is SUSTAINED (not a
    # brief blip), so retry for up to ~35s: rare, the prefetch buffer hides the stall, and waiting
    # beats losing a 38h run. Only a genuinely-broken video survives all retries -> then raise loud.
    last_err = None
    for attempt in range(40):
        try:
            loaded_frames, loaded_ts = _decode_once(video_path, timestamps)
            break
        except (av.error.BlockingIOError, av.error.ExitError, OSError) as e:  # noqa: PERF203
            last_err = e
            time.sleep(min(1.0, 0.1 * (attempt + 1)))
    else:
        raise RuntimeError(f"decode_video_frames_av failed after 40 retries (~35s) for {video_path}: {last_err}")

    query_ts = torch.tensor(timestamps)
    loaded_ts_t = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"query ts violate tolerance ({min_[~is_within_tol]} > {tolerance_s=}) in {video_path}"
    )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    return closest_frames.type(torch.float32) / 255


def patch_lerobot():
    """Monkeypatch LeRobot's decode_video_frames (imported-by-name in lerobot_dataset) with the fast av path."""
    import lerobot.common.datasets.lerobot_dataset as lds
    import lerobot.common.datasets.video_utils as vu

    vu.decode_video_frames = decode_video_frames_av
    lds.decode_video_frames = decode_video_frames_av
