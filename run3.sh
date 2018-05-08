ID=stackatt3_all
rm -r saved_models/$ID
CUDA_VISIBLE_DEVICES=3 python main.py \
    --output saved_models/$ID \
    --epochs 40 \
    --pair_loss_weight 0 \
    --pair_loss_type @att \
    --gamma 2.5 \
    --model stackatt \
    --stackatt_nlayers 3 \
    --train_dataset all \
    --test_dataset all \
    --seed 5293
