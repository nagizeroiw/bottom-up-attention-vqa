ID=pair_wise
rm -r saved_models/$ID/
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output saved_models/$ID/ \
    --epochs 40 \
    --pair_loss_weight 0
