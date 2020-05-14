#sketchy module
export DATA_DIR=data
export TASK_NAME=squad
python ./examples/run_cls.py \
    --model_type albert \
    --model_name_or_path albert-xxlarge-v2 \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --max_seq_length 512 \
    --per_gpu_train_batch_size=6   \
    --per_gpu_eval_batch_size=8   \
    --warmup_steps=814 \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --eval_all_checkpoints \
    --output_dir squad/cls_squad2_albert-xxlarge-v2_lr2e-5_len512_bs48_ep2_wm814_fp16 \
    --save_steps 2500 \
    --fp16