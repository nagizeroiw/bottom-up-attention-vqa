import json
import sys
import os
import random

image_path = {
    'train': 'train2014/COCO_train2014_000000',
    'valid': 'valid2014/COCO_val2014_000000'
}

if __name__ == '__main__':
    split = sys.argv[1]
    i_id = sys.argv[2]
    
    file_name = image_path[split] + '%06d.jpg' % int(i_id)
    print('file name:', file_name)

    os.system('scp %s ./vis/' % ('jungpu5:~/vqa-butd/data/images/%s' % file_name))
