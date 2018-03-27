ID=pair_loss
rm -r saved_models/$IED
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output saved_models/$ID/ \
    --epochs 40 \
    --pair_loss_weight 1e-3
