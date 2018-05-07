ID=dualatt_ploss
rm -r saved_models/$ID
CUDA_VISIBLE_DEVICES=2 python main.py \
    --output saved_models/$ID \
    --epochs 40 \
    --pair_loss_weight 0.05 \
    --pair_loss_type margin@repr \
    --gamma 2.5 \
    --model dualatt \
    --train_dataset pairwise \
    --test_dataset all \
    --seed 5293
