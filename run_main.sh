export DATA_DIR=./data/diverse_data
export OUTPUT_DIR=./output_main
export EVALUATE_DIR=./data
export CUDA_VISIBLE_DEVICES=0,1,2

 
python -m torch.distributed.launch --nproc_per_node=3 --master_port 29522 run_squad.py \
        --model_type bert \
        --model_name_or_path bert-large-uncased-whole-word-masking \
        --do_train \
        --do_eval \
        --do_lower_case \
        --train_file $DATA_DIR/uqa_train_main.json \
        --predict_file $EVALUATE_DIR/dev-v1.1.json \
        --learning_rate 3e-5 \
        --hidden_size 1024 \
        --num_train_epochs 2 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --output_dir $OUTPUT_DIR \
        --per_gpu_train_batch_size=4 \
        --per_gpu_eval_batch_size=128 \
        --seed 42 \
        --overwrite_output_dir \
        --gradient_accumulation_steps 2\
        --logging_steps 1000 \
        --save_steps 1000 \
        --eval_all_checkpoints \
        --max_step 5000
