ID1=test_pair_loss_4_back
ID2=test_pair_loss_4_back_w0.05
rm -r ./tf_log/
mkdir ./tf_log/
scp -r jungpu5:~/vqa-butd/saved_models/$ID1/tf_log ./tf_log/$ID1
scp -r jungpu5:~/vqa-butd/saved_models/$ID2/tf_log ./tf_log/$ID2
tensorboard --logdir=./tf_log/
