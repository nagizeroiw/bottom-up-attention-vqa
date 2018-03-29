rm -r ./tf_log/
scp -r jungpu5:~/vqa-butd/saved_models/pair_loss_2/tf_log ./
tensorboard --logdir=./tf_log/
