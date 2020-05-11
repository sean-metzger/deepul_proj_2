print('Sko Buffs')
import subprocess
import shlex
import os 


def find_model(name, epochs, basepath="/userdata/smetzger/all_deepul_files/ckpts"):
    """
    name = model name
    fold = which fold of the data to find. 
    epochs = how many epochs to load the checkpoint at (e.g. 750)
    
    """

    path_list = []
    file_list = []
    for file in os.listdir(basepath):
        if name in str(file):
            if str(file).endswith(str(epochs-1) + '.tar'): 
                path_list.append(os.path.join(basepath, file))
                file_list.append(file)
            
    return path_list, file_list

base_name = '100epochs_512bsz_0.4000lr_mlp_cos_custom_aug_single_aug_study_'
checkpoint_fp = '/userdata/smetzger/all_deepul_files/ckpts'

models, files = find_model(base_name, 100)

print(models)
print(len(models))
for model, name in zip(models, files): 
    filename = '/userdata/smetzger/all_deepul_files/runs/lincls_' + name + '_lincls.txt'
    string = "submit_job -q mind-gpu"
    string += " -m 318 -g 4"
    string += " -o " + filename
    string += ' -n kf_lincls'
    string += ' -x python /userdata/smetzger/all_deepul_files/deepul_proj/moco/main_lincls.py'

    # add all the default args: 
    string += " -a resnet50 --lr 15.0  --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1"
    string += ' --checkpoint_fp ' + str(checkpoint_fp)
    string += ' --rank 0'
    string += ' --pretrained ' + model
    string += " --data /userdata/smetzger/data/cifar_10/ --notes 'training_single_aug'"
    string += " --task rotation"
    string += " --schedule 10 20 --epochs 50"

    cmd = shlex.split(string)
    print(cmd)
    subprocess.run(cmd, stderr=subprocess.STDOUT)