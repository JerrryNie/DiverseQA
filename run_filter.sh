# use filter data + substring(NE) data
export DATA_DIR=./data/diverse_data
export EVALUATE_DIR=./data
export MAIN_MODEL_DIR=./output_main/checkpoint-2000
export OUTPUT_DIR=./output_main_filter
export CUDA_VISIBLE_DEVICES=0,1,2
export MKL_THREADING_LAYER=GNU

python multi_turn.py \
      --refine_data_dir $DATA_DIR \
      --output_dir $OUTPUT_DIR \
      --model_dir $MAIN_MODEL_DIR \
      --predict_file $EVALUATE_DIR/dev-v1.1.json \
      --score_threshold 0.1 \
      --threshold_rate 0.9 \
      --seed 17 \
      --master_port 9050 \
      --metric exact_match \
      --load_from_break_point \
      --top 1
