ID=pair_loss_4_g2.5w0.07
rm -r saved_models/$ID
CUDA_VISIBLE_DEVICES=2 python main.py \
    --output saved_models/$ID \
    --epochs 40 \
    --pair_loss_weight 0.07 \
    --pair_loss_type margin@repr \
    --gamma 2.5
