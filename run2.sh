ID=test_pair_loss_4
rm -r saved_models/$ID
CUDA_VISIBLE_DEVICES=2 python main.py \
    --output saved_models/$ID \
    --epochs 40 \
    --pair_loss_weight 0 \
    --pair_loss_type margin@repr \
    --gamma 2.5
