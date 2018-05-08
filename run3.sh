ID=dualatt_all_pair
rm -r saved_models/$ID
CUDA_VISIBLE_DEVICES=3 python main.py \
    --output saved_models/$ID \
    --epochs 40 \
    --pair_loss_weight 0 \
    --pair_loss_type @att \
    --gamma 2.5 \
    --model dualatt \
    --stackatt_nlayers 3 \
    --train_dataset all_pair \
    --test_dataset all \
    --seed 5293
