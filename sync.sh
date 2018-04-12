ID=test_pair_loss_4_lr0.003
rm -r ./tf_log/
scp -r jungpu6:~/vqa-butd/saved_models/$ID/tf_log ./
tensorboard --logdir=./tf_log/
