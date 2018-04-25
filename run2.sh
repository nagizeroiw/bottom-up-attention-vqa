ID=dualatt_all
rm -r saved_models/$ID
CUDA_VISIBLE_DEVICES=2 python main.py \
    --output saved_models/$ID \
    --epochs 40 \
    --pair_loss_weight 0 \
    --pair_loss_type @att \
    --gamma 2.5 \
    --model dualatt \
    --no-use_pair \
    --no-filter_pair
