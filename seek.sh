QID1=42000
QID2=42001

mkdir 'fig_'$QID1'_'$QID2

CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_qid $QID1 \
    --seek_output 'fig_'$QID1'_'$QID2'/'$QID1'_filter.png' \
    --start_with saved_models/dualatt_filter \
    --seed 5293

CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_qid $QID2 \
    --seek_output 'fig_'$QID1'_'$QID2'/'$QID2'_filter.png' \
    --start_with saved_models/dualatt_filter \
    --seed 5293

CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_qid $QID1 \
    --seek_output 'fig_'$QID1'_'$QID2'/'$QID1'_pairwise.png' \
    --start_with saved_models/dualatt_pairwise \
    --seed 5293

CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_qid $QID2 \
    --seek_output 'fig_'$QID1'_'$QID2'/'$QID2'_pairwise.png' \
    --start_with saved_models/dualatt_pairwise \
    --seed 5293

CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_qid $QID1 \
    --seek_output 'fig_'$QID1'_'$QID2'/'$QID1'_ploss.png' \
    --start_with saved_models/dualatt_ploss \
    --seed 5293
    
CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_qid $QID2 \
    --seek_output 'fig_'$QID1'_'$QID2'/'$QID2'_ploss.png' \
    --start_with saved_models/dualatt_ploss \
    --seed 5293
    
