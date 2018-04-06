ID=pair_loss_3_lr0.05
rm -r saved_models/$ID
CUDA_VISIBLE_DEVICES=2 python main.py \
    --output saved_models/$ID \
    --epochs 40 \
    --pair_loss_weight 0.05 \
    --pair_loss_type margin
