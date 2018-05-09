CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_qid 267225012 \
    --start_with saved_models/dualatt_filter \
    --seed 5293
