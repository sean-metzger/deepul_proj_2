print('Sko Buffs')
import subprocess
import shlex
import os 

base_model_name = ''
epochs = 750
import os
checkpoint_fp = '/userdata/smetzger/all_deepul_files/ckpts'

# For resuming 

def find_model(name, fold, epochs, basepath="/userdata/smetzger/all_deepul_files/ckpts"):
    """
    name = model name
    fold = which fold of the data to find. 
    epochs = how many epochs to load the checkpoint at (e.g. 750)
    
    """
    for file in os.listdir(basepath):
        print(file)
        if name in str(file) and 'fold_%d' %fold in str(file):
            print(file)
            if str(file).endswith(str(epochs-1) + '.tar') or str(file).endswith(str(epochs) + '.tar'): 
                return os.path.join(basepath, file)
            
    print("COULDNT FIND MODEL")
    assert True==False # just throw and error. 

base_name = '750epochs_512bsz_0.4000lr_mlp_cos_rotnet'
# Notes: This is the setup used to get the 5 folds of the rotnet for our evaluation of rotation predictions


custom_augs = ['imagenet_min_icl']
for custom_aug_name in custom_augs: 
    filename = '/userdata/smetzger/all_deepul_files/runs/imgnet_big_run_justrrc_10.txt' #%(custom_aug_name)
    string = "submit_job -q mind-gpu@mind3"
    string += " -m 318 -g 4"
    string += " -o " + filename
    string += ' -n inet_moco'
    string += ' -x python /userdata/smetzger/all_deepul_files/deepul_proj/moco/main_moco.py'

    # add all the default args: 
    string += " -a resnet50 --lr 0.015  --batch-size 128 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1"
    string += ' --moco-t 0.2' # MoCov2 arguments. 
    string += ' --checkpoint_fp ' + str(checkpoint_fp)
    string += ' --rank 0'
    string += " --data /userdata/smetzger/data/imagenet/imagenet12/ --notes 'imagenet rrc only'"
    string += ' --dataid imagenet'
    string += ' --mlp --cos --epochs 10'
    string += ' --rand_resize_only'
    # string += ' --checkpoint-interval 10'
    # string += ' --custom_aug_name ' + custom_aug_name # REDUCED IMAGENET

    cmd = shlex.split(string)
    print(cmd)
    subprocess.run(cmd, stderr=subprocess.STDOUT)