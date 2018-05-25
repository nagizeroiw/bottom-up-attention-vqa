
mkdir 'fig_sal'

CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_output './fig_sal' \
    --start_with saved_models/dualatt_filter \
    --seed 5293

CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_output './fig_sal' \
    --start_with saved_models/dualatt_pairwise \
    --seed 5293

CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_output './fig_sal' \
    --start_with saved_models/dualatt_ploss \
    --seed 5293
