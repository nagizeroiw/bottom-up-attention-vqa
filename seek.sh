QID1=153249005
QID2=30198010

CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_qid $QID1 \
    --seek_output $QID1_filter.png \
    --start_with saved_models/dualatt_filter \
    --seed 5293

CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_qid $QID2 \
    --seek_output $QID2_filter.png \
    --start_with saved_models/dualatt_filter \
    --seed 5293

CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_qid $QID1 \
    --seek_output $QID1_pairwise.png \
    --start_with saved_models/dualatt_pairwise \
    --seed 5293

CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_qid $QID2 \
    --seek_output $QID2_pairwise.png
    --start_with saved_models/dualatt_pairwise \
    --seed 5293

CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_qid $QID1 \
    --seek_output $QID1_pairwise.png
    --start_with saved_models/dualatt_ploss \
    --seed 5293
    
CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task seek-val \
    --seek_qid $QID2 \
    --seek_output $QID2_pairwise.png
    --start_with saved_models/dualatt_ploss \
    --seed 5293
    