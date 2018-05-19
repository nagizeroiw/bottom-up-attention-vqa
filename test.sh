ID=ens_tv_2345
CUDA_VISIBLE_DEVICES=3 python main.py \
    --batch_size 256 \
    --model dualatt \
    --task test-dev \
    --start_with saved_models/$ID \
    --test_output $ID
