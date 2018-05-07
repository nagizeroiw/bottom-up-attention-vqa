CUDA_VISIBLE_DEVICES=1 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task test \
    --start_with saved_models/dualatt_all
    --seed 5293
