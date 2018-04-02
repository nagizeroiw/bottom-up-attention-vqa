ID=pair_loss_3
rm -r saved_models/$ID/
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output saved_models/$ID/ \
    --epochs 40 \
    --pair_loss_weight 0 \
    --task measure \
    --start_with saved_models/pair_loss_t2
