# Source patches applied (reproduced by `apply_patches.sh`)

All are RoboPRO/curobo integration fixes for **curobo v0.7.7** (the version
RoboTwin/RoboPRO vendors). They are idempotent and verified via curobo's exact
`MotionGenConfig.load_from_robot_config` path.

### 1. curobo `world_mesh.py` — `clear_cache`
File: `customized_robotwin/envs/curobo/src/curobo/geom/sdf/world_mesh.py`
RoboPRO README Step 5. Rebuild the `_env_mesh_names` reset as an explicit
nested loop instead of a list-comprehension reassignment (matches RoboPRO's
required patch).

### 2. RoboPRO `planner.py` — `detach_object`
File: `customized_robotwin/envs/robot/planner.py`
RoboPRO calls `detach_object()` on **every** robot reset, expecting a no-op
when nothing is attached. curobo ≥0.7.7 raises
`ValueError("attached_object not found in spheres")` there (older curobo was a
silent no-op). Wrapped in `try/except ValueError` that swallows only the
"not found in spheres" case.

### 3–5. curobo embodiment configs (`aloha-agilex`)
Files: `benchmark/assets/embodiments/aloha-agilex/curobo_{left,right}.yml`
and the `*_tmp.yml` templates (so `update_embodiment_config_path.py`
regeneration keeps them; templates are what survive an asset re-download).

curobo v0.7.7 requires the `attached_object` attach link to be **fully
registered** before `attach_external_objects_to_robot` is called. The shipped
configs had it missing/empty, causing `KeyError: 'attached_object'`. Three
coordinated edits:

| Edit | From | To |
|---|---|---|
| `extra_collision_spheres` | `{}` | `{attached_object: 100}` |
| `extra_links` | `null` | `attached_object` link: `parent_link_name` = `fl_link6` (left) / `fr_link6` (right), `joint_type: FIXED`, identity `fixed_transform` |
| `collision_link_names` | (list) | append `"attached_object"` |

Together these make curobo register `attached_object` with 100 spheres
(verified: `get_number_of_spheres("attached_object") == 100`), which is what
unblocked RoboPRO expert data collection (3/3 smoke episodes succeeded).

### Not a curobo patch, but required: ffmpeg on PATH
`pkl → hdf5` conversion shells out to `ffmpeg`. The robotwin env ships an
ffmpeg binary in its `bin/`; `setup_server.sh` installs conda-forge ffmpeg and
collection must run with the env `bin/` on `PATH` (the run scripts do this).

### DA3-XVLA application-side changes
These are first-class source in `DA3-XVLA/` (not "patches") and travel with the
rsync: `models/geometry_conditioning.py` (DA3LatentSegmenter, resampler,
fusion), `models/da3_wrapper.py` (real DA3 latent extraction),
`models/modeling_xvla.py` / `transformer.py` (fusion + finetune policy),
`models/configuration_xvla.py` (config schema), `datasets/da3_loader.py` +
dataset hooks, `scripts/` (precompute_*, make_robotwin_meta,
debug_geometry_conditioning), `tests/`, `docs/`.
