ID=pair_loss_1
rm -r ./tf_log/
scp -r jungpu5:~/vqa-butd/saved_models/$ID/tf_log ./
tensorboard --logdir=./tf_log/
