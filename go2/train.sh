#!/usr/bin/env bash
# Train / play a Unitree Go2 velocity-tracking policy with Isaac Lab (RSL-RL PPO).
#
#   ~/robots_isaac/go2/train.sh train            # flat terrain, 4096 envs, headless
#   ~/robots_isaac/go2/train.sh train rough      # rough terrain (longer)
#   ~/robots_isaac/go2/train.sh train flat --gui --num_envs 64   # train with the viewport
#   ~/robots_isaac/go2/train.sh play             # watch the latest flat checkpoint
#   ~/robots_isaac/go2/train.sh play rough
#
# Extra args are passed through to train.py / play.py, e.g.:
#   ~/robots_isaac/go2/train.sh train flat --num_envs 2048 --max_iterations 500 --seed 1
set -euo pipefail

MODE="${1:-train}"; shift || true
TERRAIN="${1:-flat}"
case "$TERRAIN" in flat|rough) shift || true ;; *) TERRAIN=flat ;; esac

T="$(tr '[:lower:]' '[:upper:]' <<< "${TERRAIN:0:1}")${TERRAIN:1}"
TASK="Isaac-Velocity-${T}-Unitree-Go2-v0"
RL=~/IsaacLab/scripts/reinforcement_learning/rsl_rl

# --gui: open the viewport while training (slower; use fewer envs, e.g. --num_envs 64)
HEADLESS="--headless"
ARGS=()
for a in "$@"; do [ "$a" = "--gui" ] && HEADLESS="" || ARGS+=("$a"); done
set -- "${ARGS[@]+"${ARGS[@]}"}"

cd ~/IsaacLab
case "$MODE" in
  train) exec ./isaaclab.sh -p "$RL/train.py" --task "$TASK" --num_envs 4096 $HEADLESS "$@" ;;
  play)  exec ./isaaclab.sh -p "$RL/play.py" --task "${TASK%-v0}-Play-v0" --num_envs 32 "$@" ;;
  *) echo "usage: $0 {train|play} [flat|rough] [extra args]"; exit 1 ;;
esac
