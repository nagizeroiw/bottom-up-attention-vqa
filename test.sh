CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task test-dev \
    --start_with saved_models/dualatt_all \
    --seed 5293
