ID=pair_loss_1
rm -r saved_models/$ID/
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --output saved_models/$ID/ \
    --epochs 40 \
    --pair_loss_weight 0 \
    --pair_loss_type @repr
