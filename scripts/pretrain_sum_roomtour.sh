#!/bin/bash 
# THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
# cd $THIS_DIR
# config=$1
# ckpt_path=$2
# n_gpu=$2
echo $METIS_WORKER_0_HOST
echo $METIS_WORKER_0_PORT
echo $METIS_TASK_INDEX

ports=(${METIS_WORKER_0_PORT//,/ })
port=${ports[2]}

# set mp3d path
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH

# set java path
export JAVA_HOME=$java_path
export PATH=$JAVA_HOME/bin:$PATH
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar


torchrun --nproc_per_node=8 --nnode=$NNODE --node_rank=$NODE_ID --master_addr=$METIS_WORKER_0_HOST --master_port=$port pretrain.py \
    --stage pretrain --cfg_file configs/pretrain.yaml \
    --data_dir /home/tiger/VLN --pretrained_model_name_or_path /mnt/bn/kinetics-lp-maliva-v6/pretrain_models/vicuna-7b-v1.1-from-delta/ --precision amp_bf16 \
    --batch_size 1 --gradient_accumulation_step 8 --num_steps_per_epoch 500 --lr 1e-4 --seed 0 --num_epochs 60 \
    --enable_og --enable_summarize --enable_fgr2r \
    --test_datasets CVDN SOON R2R REVERIE ScanQA \
    --max_saved_checkpoints 15 --output_dir output/pretrain_w_aug_w_room_add_objroomtype_tour3d10_1e-4_60e_re_fix \
    --save_latest_states
