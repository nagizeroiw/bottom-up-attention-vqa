CUDA_VISIBLE_DEVICES=0 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task test \
    --start_with saved_models/dualatt_pairwise \
    --seed 5293
