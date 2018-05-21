ID=stacksameatt3
rm -r saved_models/$ID
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output saved_models/$ID \
    --epochs 40 \
    --pair_loss_weight 0 \
    --pair_loss_type @att \
    --gamma 2.5 \
    --model dualatt \
    --train_dataset trainval \
    --test_dataset all \
    --seed 5555
