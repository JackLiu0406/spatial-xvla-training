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

import torch
from torch.utils.data import DataLoader
from .dataset import InfiniteDataReader

def worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % (2**32)
    import random, numpy as np
    np.random.seed(base_seed); random.seed(base_seed); torch.manual_seed(base_seed)


def create_dataloader(batch_size: int,
                      metas_path: str,
                      num_actions: int,
                      training: bool,
                      action_mode: str,
                      geometry_conditioning: dict | None = None,
                      ):
    import os
    # XVLA_NUM_WORKERS: per-rank dataloader workers. With 8 ranks × N workers,
    # 224 CPU slots can support up to ~24 workers/rank cleanly. Default 8 keeps
    # back-compat with older runs; bump to 16-24 if step time is dataloader-bound
    # (look for high step-time variance and low GPU utilization between iters).
    _nw = int(os.environ.get("XVLA_NUM_WORKERS", "8"))
    _pf = int(os.environ.get("XVLA_PREFETCH_FACTOR", "2"))
    return DataLoader(
        InfiniteDataReader(metas_path, num_actions=num_actions, training=training,
                           action_mode=action_mode,
                           geometry_conditioning=geometry_conditioning),
        batch_size=batch_size,
        num_workers=_nw,
        prefetch_factor=_pf,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )