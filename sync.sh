ID1=dualatt_2layer
ID2=dualatt_no_pair
ID3=dualatt_1280
scp -r jungpu6:~/vqa-butd/saved_models/$ID1/tf_log ./tf_log/$ID1
scp -r jungpu6:~/vqa-butd/saved_models/$ID2/tf_log ./tf_log/$ID2
scp -r jungpu6:~/vqa-butd/saved_models/$ID3/tf_log ./tf_log/$ID3
tensorboard --logdir=./tf_log/
