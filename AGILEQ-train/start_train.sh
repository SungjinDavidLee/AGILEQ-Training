#!/bin/bash
export CARLA_ROOT="/home/user/yechan_work/carla/CARLA_0.9.15/"


export HOST=localhost
export PORT=2000
export TOTAL_TIMESTEPS=600000
export RELOAD_MODEL=""
export FPS=15
export NUM_CHECKPOINTS=12
export CONFIG="crossq_pp"
export TOWN="town01"
export SAVE_DIR="./results"

python3 train.py \
--host=${HOST} \
--port=${PORT} \
--total_timesteps=${TOTAL_TIMESTEPS} \
--reload_model=${RELOAD_MODEL} \
--fps=${FPS} \
--num_checkpoints=${NUM_CHECKPOINTS} \
--config=${CONFIG} \
--save_dir=${SAVE_DIR} \
--town=${TOWN} \
--no_render
