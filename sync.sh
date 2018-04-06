ID=pair_loss_3
rm -r ./tf_log/
scp -r jungpu6:~/vqa-butd/saved_models/$ID/tf_log ./
tensorboard --logdir=./tf_log/
