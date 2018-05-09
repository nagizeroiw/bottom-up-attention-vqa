ID1=dualatt_all_pair
scp -r jungpu6:~/vqa-butd/saved_models/$ID1/tf_log ./tf_log/$ID1
tensorboard --logdir=./tf_log/
