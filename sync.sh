rm -r ./tf_log/
scp -r jungpu5:~/vqa-butd/saved_models/pair_wise/tf_log ./
tensorboard --logdir=./tf_log/
