ID=test_pair_loss_4_back_w0.1
rm -r saved_models/$ID
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output saved_models/$ID \
    --epochs 40 \
    --pair_loss_weight 0.1 \
    --pair_loss_type margin@repr \
    --gamma 2.5 \
    --seed 5293
