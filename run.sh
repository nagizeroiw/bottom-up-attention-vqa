ID=test_resnet
rm -r saved_models/$ID
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --output saved_models/$ID \
    --epochs 40 \
    --pair_loss_weight 0 \
    --pair_loss_type @att \
    --gamma 2.5 \
    --batch_size 256 \
    --model fine \
    --train_dataset end2end \
    --test_dataset end2end \
    --seed 5293
