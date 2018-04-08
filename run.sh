ID=pair_loss_4_g3
rm -r saved_models/$ID
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output saved_models/$ID \
    --epochs 40 \
    --pair_loss_weight 0.05 \
    --pair_loss_type margin@repr \
    --gamma 3
