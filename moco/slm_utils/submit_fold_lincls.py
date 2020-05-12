print('Sko Buffs')
import subprocess
import shlex
import os 

base_model_name = ''
epochs = 750
import os

def find_model(name, fold, epochs, basepath="/userdata/smetzger/all_deepul_files/ckpts"):
    """
    name = model name
    fold = which fold of the data to find. 
    epochs = how many epochs to load the checkpoint at (e.g. 750)
    
    """

    path_list = []

    for file in os.listdir(basepath):
        if name in str(file) and 'svhn' in str(file):
            if str(file).endswith(str(epochs-1) + '.tar'): 
                if 'fold_%d' %(fold) in file: 
                    return (os.path.join(basepath, file))
            
    print("COULDNT FIND MODEL")
    assert True==False # just throw and error. 

base_name = '750epochs_512bsz_0.4000lr_mlp_cos_fold'
checkpoint_fp = '/userdata/smetzger/all_deepul_files/ckpts'
epochs = 750

i = 0

for task in ['rotation', 'classify']: 
    for fold in range (5): 
        filename = '/userdata/smetzger/all_deepul_files/runs/lincls' + base_name  + '_fold_%d' %fold + '_' + task + '_kfold.txt'
        string = "submit_job -q mind-gpu"
        string += " -m 318 -g 4"
        string += " -o " + filename
        string += ' -n kf_lincls'
        string += ' -x python /userdata/smetzger/all_deepul_files/deepul_proj/moco/main_lincls.py'

        # add all the default args: 
        string += " -a resnet50 --lr 15.0  --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1"
        string += ' --checkpoint_fp ' + str(checkpoint_fp)
        string += ' --rank 0'
        string += ' --pretrained ' + str(find_model(base_name, fold, 750))
        string += " --data /userdata/smetzger/data/cifar_10/ --notes 'training_lincls_on_just RRC svhn'"
        string += " --kfold %d" %fold
        string += " --task " + task
        string += " --dataid svhn"
        string += " --schedule 10 20 --epochs 50"

        cmd = shlex.split(string)
        print(cmd)
        i+=1
        print(i)
        subprocess.run(cmd, stderr=subprocess.STDOUT)