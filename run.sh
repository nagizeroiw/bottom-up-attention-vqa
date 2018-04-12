ID=pair_loss_5_g2.5
rm -r saved_models/$ID
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output saved_models/$ID \
    --epochs 40 \
    --pair_loss_weight 0.05 \
    --pair_loss_type margin@jrepr \
    --gamma 2.5
