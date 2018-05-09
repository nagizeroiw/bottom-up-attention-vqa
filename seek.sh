CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --start_with saved_models/dualatt_ploss \
    --seed 5293
