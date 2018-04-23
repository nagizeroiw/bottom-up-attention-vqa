ID=dualatt_no_pair
rm -r saved_models/$ID
CUDA_VISIBLE_DEVICES=3 python main.py \
    --output saved_models/$ID \
    --epochs 40 \
    --use_pair False \
    --model dualatt
