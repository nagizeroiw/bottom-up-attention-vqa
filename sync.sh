ID1=dualatt_ploss3
ID2=dualatt_ploss3
ID3=dualatt_withploss4
scp -r jungpu6:~/vqa-butd/saved_models/$ID1/tf_log ./tf_log/$ID1
scp -r jungpu6:~/vqa-butd/saved_models/$ID2/tf_log ./tf_log/$ID2
scp -r jungpu6:~/vqa-butd/saved_models/$ID3/tf_log ./tf_log/$ID3
tensorboard --logdir=./tf_log/
