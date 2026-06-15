#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

"""
Build an X-VLA meta JSON from RoboPRO/RoboTwin-collected demonstrations.

RoboPRO ``collect_data.py`` writes:

    <data_root>/<task>/<config>/data/episode{i}.hdf5
    <data_root>/<task>/<config>/instructions/episode{i}.json   {"seen":[...],"unseen":[...]}

X-VLA's ``RobotWin2Handler`` (dataset_name pattern ``robotwin2-*``) consumes
HDF5 with:

    endpose/left_endpose   [T,7]   xyz(3)+quat(4)
    endpose/right_endpose  [T,7]
    endpose/left_gripper   [T]
    endpose/right_gripper  [T]
    observation/<cam>/rgb  [T] (encoded jpg)   <- cameras auto-detected

and reads the language string from ``meta["language_instruction_key"]`` *inside
the HDF5*. RoboTwin stores instructions in sibling JSONs, so (unless
--no-inject) we write a scalar ``instruction`` dataset into each episode HDF5
picked from the chosen split.

This script: validates the schema, (optionally) injects instructions, and
emits the meta JSON in the "general style" ``InfiniteDataReader`` expects.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import h5py
import numpy as np


def _episode_idx(fn: str) -> int:
    # Strip the extension first so e.g. ".hdf5" digits don't bleed into the idx
    # (episode0.hdf5 -> 0, not 05).
    stem = os.path.splitext(os.path.basename(fn))[0]
    s = "".join(c for c in stem if c.isdigit())
    return int(s) if s else -1


def _detect_cameras(f: h5py.File) -> list:
    if "observation" not in f:
        return []
    cams = []
    for cam in f["observation"].keys():
        g = f["observation"][cam]
        if isinstance(g, h5py.Group) and "rgb" in g:
            cams.append(f"observation/{cam}/rgb")
    return sorted(cams)


def _validate(path: str) -> tuple:
    """Return (ok, report_str, cameras, T)."""
    try:
        with h5py.File(path, "r") as f:
            need = ["endpose/left_endpose", "endpose/right_endpose",
                    "endpose/left_gripper", "endpose/right_gripper"]
            missing = [k for k in need if k not in f]
            if missing:
                return False, f"missing keys {missing}", [], 0
            le = f["endpose/left_endpose"]
            re = f["endpose/right_endpose"]
            if le.shape[-1] != 7 or re.shape[-1] != 7:
                return (False,
                        f"endpose last dim {le.shape[-1]}/{re.shape[-1]} != 7 "
                        "(RobotWin2Handler expects xyz+quat)", [], 0)
            cams = _detect_cameras(f)
            if not cams:
                return False, "no observation/<cam>/rgb cameras", [], 0
            T = le.shape[0]
            return True, f"OK T={T} cams={len(cams)}", cams, T
    except Exception as e:
        return False, f"open failed: {e}", [], 0


def _pick_instruction(instr_dir: str, idx: int, split: str) -> str | None:
    p = os.path.join(instr_dir, f"episode{idx}.json")
    if not os.path.isfile(p):
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    lst = d.get(split) or d.get("seen") or d.get("unseen") or []
    return random.choice(lst) if lst else None


def _inject(path: str, instruction: str, key: str) -> None:
    with h5py.File(path, "a") as f:
        if key in f:
            del f[key]
        # scalar bytes -> RobotWin2Handler.read_instruction does v.decode()
        f.create_dataset(key, data=np.bytes_(instruction.encode("utf-8")))


def get_args():
    p = argparse.ArgumentParser("RoboPRO -> X-VLA meta generator")
    p.add_argument("--data-root", required=True,
                   help="dir containing <task>/<config>/data/episode*.hdf5")
    p.add_argument("--task", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True, help="output meta .json path")
    p.add_argument("--dataset-name", default="",
                   help="default: robotwin2-<task>")
    p.add_argument("--camera-keys", default="",
                   help="comma list; default = auto-detected, sorted")
    p.add_argument("--instruction-split", default="seen",
                   choices=["seen", "unseen"])
    p.add_argument("--language-instruction-key", default="instruction")
    p.add_argument("--no-inject", action="store_true",
                   help="don't write instruction into HDF5 (assume present)")
    p.add_argument("--default-instruction", default="",
                   help="fallback when no instruction JSON is found")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    base = os.path.join(args.data_root, args.task, args.config)
    data_dir = os.path.join(base, "data")
    instr_dir = os.path.join(base, "instructions")
    if not os.path.isdir(data_dir):
        sys.exit(f"[error] no data dir: {data_dir} (run collect_data first)")

    eps = sorted((os.path.join(data_dir, fn) for fn in os.listdir(data_dir)
                  if fn.endswith(".hdf5")), key=_episode_idx)
    if not eps:
        sys.exit(f"[error] no episode*.hdf5 in {data_dir}")

    dataset_name = args.dataset_name or f"robotwin2-{args.task}"
    forced_cams = [c for c in args.camera_keys.split(",") if c]

    datalist, cams_ref, kept, skipped = [], None, 0, 0
    for path in eps:
        ok, report, cams, T = _validate(path)
        idx = _episode_idx(path)
        if not ok:
            print(f"[skip] episode{idx}: {report}")
            skipped += 1
            continue
        use_cams = forced_cams or cams
        if cams_ref is None:
            cams_ref = use_cams
        elif use_cams != cams_ref:
            print(f"[skip] episode{idx}: camera set {use_cams} != {cams_ref}")
            skipped += 1
            continue

        if not args.no_inject:
            instr = _pick_instruction(instr_dir, idx, args.instruction_split) \
                or args.default_instruction
            if not instr:
                print(f"[skip] episode{idx}: no instruction "
                      f"(json missing & no --default-instruction)")
                skipped += 1
                continue
            _inject(path, instr, args.language_instruction_key)

        datalist.append(path)
        kept += 1
        print(f"[ok ] episode{idx}: {report}")

    if not datalist:
        sys.exit("[error] no valid episodes -> no meta written")

    meta = {
        "dataset_name": dataset_name,
        "datalist": datalist,
        "observation_key": cams_ref,
        "language_instruction_key": args.language_instruction_key,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[done] {kept} kept / {skipped} skipped")
    print(f"[meta] {args.out}")
    print(f"  dataset_name = {dataset_name}  (-> RobotWin2Handler via "
          f"'robotwin2-*')")
    print(f"  observation_key = {cams_ref}")
    print(f"  language_instruction_key = {args.language_instruction_key}")
    print("Use it as: train.py --train_metas_path "
          f"{os.path.dirname(os.path.abspath(args.out))}")


if __name__ == "__main__":
    main()
