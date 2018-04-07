ID=pair_loss_3_g2.0
rm -r ./tf_log/
scp -r jungpu6:~/vqa-butd/saved_models/$ID/tf_log ./
tensorboard --logdir=./tf_log/
