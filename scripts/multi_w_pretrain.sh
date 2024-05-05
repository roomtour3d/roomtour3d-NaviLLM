# set mp3d path
# export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
set -x

# set java path
# export JAVA_HOME=$java_path
# export PATH=$JAVA_HOME/bin:$PATH
# export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar

# activate environment
# conda activate navillm

# # training for 20 epochs
# /mnt/bn/kinetics-lp-maliva/envs/conda/envs/navillm3/bin/torchrun --nnodes=1 --nproc_per_node=8 --master_port 19923 train.py \
#     --stage multi --cfg_file configs/multi.yaml \
#     --data_dir /home/tiger/VLN --pretrained_model_name_or_path /mnt/bn/kinetics-lp-maliva-v6/pretrain_models/vicuna-7b-v1.1-from-delta/ --precision amp_bf16 \
#     --resume_from_checkpoint output/pretrain_default/pretrain_39.pt \
#     --batch_size 1 --gradient_accumulation_step 8 --num_steps_per_epoch 2000 --lr 3e-5 --seed 0 --num_epochs 10 \
#     --enable_og --enable_summarize --enable_fgr2r \
#     --test_datasets CVDN SOON R2R REVERIE ScanQA \
#     --max_saved_checkpoints 1 --output_dir output/multi_pretrain_vicuna1.1_default_lr1e-4 \
#     --resume_from_checkpoint output/multi_pretrain_vicuna1.1_default_lr1e-4/epoch_9.pt \
#     --teacher_forcing_coef 1 --save_latest_states

# /mnt/bn/kinetics-lp-maliva/envs/conda/envs/navillm3/bin/torchrun --nnodes=1 --nproc_per_node=8 --master_port 19923 train.py \
#     --stage multi --cfg_file configs/multi.yaml \
#     --data_dir /home/tiger/VLN --pretrained_model_name_or_path /mnt/bn/kinetics-lp-maliva-v6/pretrain_models/vicuna-7b-v1.1-from-delta/ --precision amp_bf16 \
#     --resume_from_checkpoint output/pretrain_w_aug_w_room_only_objroomtype_1e-4_40e_fixed/pretrain_39.pt \
#     --batch_size 1 --gradient_accumulation_step 8 --num_steps_per_epoch 2000 --lr 3e-5 --seed 0 --num_epochs 20 \
#     --enable_og --enable_summarize --enable_fgr2r \
#     --test_datasets CVDN SOON R2R REVERIE ScanQA \
#     --max_saved_checkpoints 1 --output_dir output/multi_pretrain_vicuna1.1_room20_waug_wobj_wr2r_only_objroomtype_lr1e-4 \
#     --teacher_forcing_coef 1 --save_latest_states

# /mnt/bn/kinetics-lp-maliva/envs/conda/envs/navillm3/bin/torchrun --nnodes=1 --nproc_per_node=8 --master_port 19923 train.py \
#     --stage multi --cfg_file configs/multi_wroom.yaml \
#     --data_dir /home/tiger/VLN --pretrained_model_name_or_path /mnt/bn/kinetics-lp-maliva-v6/pretrain_models/vicuna-7b-v1.1-from-delta/ --precision amp_bf16 \
#     --resume_from_checkpoint output/pretrain_w_aug_w_room_only_objroomtype_1e-4_40e_fixed/pretrain_39.pt \
#     --batch_size 1 --gradient_accumulation_step 8 --num_steps_per_epoch 2000 --lr 3e-5 --seed 0 --num_epochs 20 \
#     --enable_og --enable_summarize --enable_fgr2r \
#     --test_datasets CVDN SOON R2R REVERIE ScanQA \
#     --max_saved_checkpoints 1 --output_dir output/multi_pretrain_vicuna1.1_room20_waug_wobj_wr2r_only_objroomtype_lr1e-4_wroom20 \
#     --teacher_forcing_coef 1 --save_latest_states


# /mnt/bn/kinetics-lp-maliva/envs/conda/envs/navillm3/bin/torchrun --nnodes=1 --nproc_per_node=8 --master_port 19923 train.py \
#     --stage multi --cfg_file configs/multi.yaml \
#     --data_dir /home/tiger/VLN --pretrained_model_name_or_path /mnt/bn/kinetics-lp-maliva-v6/pretrain_models/vicuna-7b-v1.1-from-delta/ --precision amp_bf16 \
#     --resume_from_checkpoint output/pretrain_w_aug_w_room_only_objroomtype_1e-4_40e_re_fix/pretrain_9.pt \
#     --batch_size 1 --gradient_accumulation_step 8 --num_steps_per_epoch 2000 --lr 3e-5 --seed 0 --num_epochs 2 \
#     --enable_og --enable_summarize --enable_fgr2r \
#     --test_datasets CVDN SOON R2R REVERIE ScanQA \
#     --max_saved_checkpoints 1 --output_dir output/multi_pretrain_room20_waug_wobj_wr2r_add_objroomtype_e9_lr1e-4 \
#     --teacher_forcing_coef 1 --save_latest_states

# /mnt/bn/kinetics-lp-maliva/envs/conda/envs/navillm3/bin/torchrun --nnodes=1 --nproc_per_node=8 --master_port 19923 train.py \
#     --stage multi --cfg_file configs/multi.yaml \
#     --data_dir /home/tiger/VLN --pretrained_model_name_or_path /mnt/bn/kinetics-lp-maliva-v6/pretrain_models/vicuna-7b-v1.1-from-delta/ --precision amp_bf16 \
#     --resume_from_checkpoint output/pretrain_w_aug_w_room_only_objroomtype_1e-4_40e_re_fix/pretrain_19.pt \
#     --batch_size 1 --gradient_accumulation_step 8 --num_steps_per_epoch 2000 --lr 3e-5 --seed 0 --num_epochs 2 \
#     --enable_og --enable_summarize --enable_fgr2r \
#     --test_datasets CVDN SOON R2R REVERIE ScanQA \
#     --max_saved_checkpoints 1 --output_dir output/multi_pretrain_room20_waug_wobj_wr2r_add_objroomtype_e19_lr1e-4 \
#     --teacher_forcing_coef 1 --save_latest_states

# /mnt/bn/kinetics-lp-maliva/envs/conda/envs/navillm3/bin/torchrun --nnodes=1 --nproc_per_node=8 --master_port 19923 train.py \
#     --stage multi --cfg_file configs/multi.yaml \
#     --data_dir /home/tiger/VLN --pretrained_model_name_or_path /mnt/bn/kinetics-lp-maliva-v6/pretrain_models/vicuna-7b-v1.1-from-delta/ --precision amp_bf16 \
#     --resume_from_checkpoint output/pretrain_w_aug_w_room_add_objroomtype_1e-4_40e_re_fix/pretrain_9.pt \
#     --batch_size 1 --gradient_accumulation_step 8 --num_steps_per_epoch 2000 --lr 3e-5 --seed 0 --num_epochs 2 \
#     --enable_og --enable_summarize --enable_fgr2r \
#     --test_datasets CVDN SOON R2R REVERIE ScanQA \
#     --max_saved_checkpoints 1 --output_dir output/multi_pretrain_room20_waug_wobj_wr2r_only_objroomtype_e9_lr1e-4 \
#     --teacher_forcing_coef 1 --save_latest_states

/mnt/bn/kinetics-lp-maliva/envs/conda/envs/navillm3/bin/torchrun --nnodes=1 --nproc_per_node=8 --master_port 19923 train.py \
    --stage multi --cfg_file configs/multi.yaml \
    --data_dir /home/tiger/VLN --pretrained_model_name_or_path /mnt/bn/kinetics-lp-maliva-v6/pretrain_models/vicuna-7b-v1.1-from-delta/ --precision amp_bf16 \
    --resume_from_checkpoint output/pretrain_w_aug_w_room_only_objroomtype_1e-4_40e_re_fix/pretrain_29.pt \
    --batch_size 1 --gradient_accumulation_step 8 --num_steps_per_epoch 2000 --lr 3e-5 --seed 0 --num_epochs 2 \
    --enable_og --enable_summarize --enable_fgr2r \
    --test_datasets CVDN SOON R2R REVERIE ScanQA \
    --max_saved_checkpoints 1 --output_dir output/multi_pretrain_room20_waug_wobj_wr2r_add_objroomtype_e29_lr1e-4 \
    --teacher_forcing_coef 1 --save_latest_states

/mnt/bn/kinetics-lp-maliva/envs/conda/envs/navillm3/bin/torchrun --nnodes=1 --nproc_per_node=8 --master_port 19923 train.py \
    --stage multi --cfg_file configs/multi.yaml \
    --data_dir /home/tiger/VLN --pretrained_model_name_or_path /mnt/bn/kinetics-lp-maliva-v6/pretrain_models/vicuna-7b-v1.1-from-delta/ --precision amp_bf16 \
    --resume_from_checkpoint output/pretrain_w_aug_w_room_only_objroomtype_1e-4_40e_re_fix/pretrain_39.pt \
    --batch_size 1 --gradient_accumulation_step 8 --num_steps_per_epoch 2000 --lr 3e-5 --seed 0 --num_epochs 20 \
    --enable_og --enable_summarize --enable_fgr2r \
    --test_datasets CVDN SOON R2R REVERIE ScanQA \
    --max_saved_checkpoints 1 --output_dir output/multi_pretrain_room20_waug_wobj_wr2r_add_objroomtype_e39_lr1e-4 \
    --teacher_forcing_coef 1 --save_latest_states
    
# /mnt/bn/kinetics-lp-maliva/envs/conda/envs/navillm3/bin/torchrun --nnodes=1 --nproc_per_node=8 --master_port 19923 train.py \
#     --stage multi --cfg_file configs/multi.yaml \
#     --data_dir /home/tiger/VLN --pretrained_model_name_or_path /mnt/bn/kinetics-lp-maliva-v6/pretrain_models/vicuna-7b-v1.1-from-delta/ --precision amp_bf16 \
#     --resume_from_checkpoint output/pretrain_wonly_room_1e-4_40e_fixed/pretrain_39.pt \
#     --batch_size 1 --gradient_accumulation_step 8 --num_steps_per_epoch 2000 --lr 3e-5 --seed 0 --num_epochs 20 \
#     --enable_og --enable_summarize --enable_fgr2r \
#     --test_datasets CVDN SOON R2R REVERIE ScanQA \
#     --max_saved_checkpoints 1 --output_dir output/multi_pretrain_vicuna1.1_wonly_room_lr1e-4 \
#     --teacher_forcing_coef 1 

/mnt/bn/kinetics-lp-maliva/envs/conda/envs/navillm3/bin/torchrun --nproc_per_node=8 --master_port=19327 /mnt/bn/kinetics-lp-maliva-v6/tools/occu_full.py 
