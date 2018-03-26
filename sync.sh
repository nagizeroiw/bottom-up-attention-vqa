rm -r ./tf_log/
scp -r jungpu5:~/vqa-butd/saved_models/nbaseline/tf_log ./
tensorboard --logdir=./tf_log/
