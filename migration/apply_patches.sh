#!/bin/bash
# ------------------------------------------------------------------------------
# Re-apply every source patch this project needs on top of fresh upstream
# clones. Idempotent: safe to run repeatedly. Standalone or called by
# setup_server.sh.
#
#   WORK   : project root (default /shared_work/jack)
#
# Patches:
#   1. curobo world_mesh.py  clear_cache  (RoboPRO README Step 5)
#   2. RoboPRO planner.py    detach_object -> no-op when nothing attached
#   3. curobo_{left,right}{,_tmp}.yml  extra_collision_spheres {attached_object:100}
#   4. curobo_{left,right}{,_tmp}.yml  extra_links: attached_object
#   5. curobo_{left,right}{,_tmp}.yml  collision_link_names += attached_object
# (3-5 only run if benchmark/assets/embodiments exists; otherwise warned.)
# ------------------------------------------------------------------------------
set -u
WORK="${WORK:-/shared_work/jack}"
RP="$WORK/DA3-VLA/RoboPRO"
CB="$RP/customized_robotwin/envs/curobo"
EMB="$RP/benchmark/assets/embodiments/aloha-agilex"
py() { python3 "$@"; }

echo "[patch] 1/5 curobo world_mesh.py clear_cache"
py - "$CB/src/curobo/geom/sdf/world_mesh.py" <<'PY'
import sys,io
p=sys.argv[1]
try: s=open(p).read()
except FileNotFoundError: print("  SKIP (missing):",p); sys.exit(0)
old='''        if self._env_mesh_names is not None:
            self._env_mesh_names = [
                [None for _ in range(self.cache["mesh"])] for _ in range(self.n_envs)
            ]

        super().clear_cache()'''
new='''        if self._env_mesh_names is not None:
            for i in range(self.n_envs):
                for j in range(len(self._env_mesh_names)):
                    self._env_mesh_names[i][j] = None
        super().clear_cache()'''
if "for j in range(len(self._env_mesh_names)):" in s: print("  already patched")
elif old in s: open(p,"w").write(s.replace(old,new)); print("  patched")
else: print("  WARN: clear_cache block not found verbatim (curobo version drift?)")
PY

echo "[patch] 2/5 RoboPRO planner.py detach_object"
py - "$RP/customized_robotwin/envs/robot/planner.py" <<'PY'
import sys
p=sys.argv[1]
try: s=open(p).read()
except FileNotFoundError: print("  SKIP (missing):",p); sys.exit(0)
old='''        def detach_object(self):
            for mg in [self.motion_gen, self.motion_gen_batch]:
                mg.detach_object_from_robot(link_name="attached_object",)'''
new='''        def detach_object(self):
            # curobo>=0.7.7 raises when nothing is attached; RoboPRO calls
            # this on every reset expecting a no-op. Restore that behavior.
            for mg in [self.motion_gen, self.motion_gen_batch]:
                try:
                    mg.detach_object_from_robot(link_name="attached_object",)
                except ValueError as e:
                    if "not found in spheres" not in str(e):
                        raise'''
if "not found in spheres" in s: print("  already patched")
elif old in s: open(p,"w").write(s.replace(old,new)); print("  patched")
else: print("  WARN: detach_object block not found verbatim")
PY

if [ ! -d "$EMB" ]; then
  echo "[patch] 3-5 SKIPPED: $EMB missing (download assets + run"
  echo "        update_embodiment_config_path.py first, then re-run this)."
  exit 0
fi

echo "[patch] 3-5 curobo yml (extra_collision_spheres / extra_links / collision_link_names)"
py - "$EMB" <<'PY'
import os,re,sys
emb=sys.argv[1]
specs={"curobo_left.yml":"fl_link6","curobo_left_tmp.yml":"fl_link6",
       "curobo_right.yml":"fr_link6","curobo_right_tmp.yml":"fr_link6"}
for fn,parent in specs.items():
    p=os.path.join(emb,fn)
    try: s=open(p).read()
    except FileNotFoundError: print("  SKIP (missing):",fn); continue
    orig=s
    # 3: extra_collision_spheres
    s=s.replace("extra_collision_spheres: {}",
                "extra_collision_spheres: {attached_object: 100}")
    # 4: extra_links
    if "extra_links: null" in s:
        blk=("extra_links:\n"
             "      attached_object:\n"
             f"        parent_link_name: {parent}\n"
             "        link_name: attached_object\n"
             "        fixed_transform: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]\n"
             "        joint_type: FIXED\n"
             "        joint_name: attach_joint")
        s=s.replace("extra_links: null",blk)
    # 5: collision_link_names += attached_object
    m=re.search(r"(collision_link_names:\s*\n\s*\[)(.*?)(\s*\])",s,re.S)
    if m and '"attached_object"' not in m.group(2):
        s=s[:m.start()]+m.group(1)+m.group(2).rstrip()+',\n        "attached_object"\n      '+m.group(3)+s[m.end():]
    if s!=orig: open(p,"w").write(s); print("  patched",fn)
    else: print("  already patched / nothing to do",fn)
PY
echo "[patch] done."
