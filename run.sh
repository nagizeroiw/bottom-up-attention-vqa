ID=pair_loss_2
rm -r saved_models/$ID/
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output saved_models/$ID/ \
    --epochs 40 \
    --pair_loss_weight 1e-5
