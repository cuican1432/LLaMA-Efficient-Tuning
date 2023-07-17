#!/bin/bash

pip install -r requirements.txt;


CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path cuican1432/baichuan13b_dataphant \
    --do_train \
    --padding_side right\
    --dataset ab_qa \
    --finetuning_type lora \
    --output_dir Baichuan-13B-Base_checkpoint \
    --overwrite_cache \
    --lora_target W_pack \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate 5e-5 \
    --num_train_epochs 0.5 \
    --plot_loss \
    --fp16
;