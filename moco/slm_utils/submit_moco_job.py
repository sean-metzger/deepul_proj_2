print('Sko Buffs')
import subprocess
import shlex
import os 


checkpoint_fp = '/userdata/smetzger/all_deepul_files/ckpts'


custom_aug_name = 'moco_rotation_top1'
 
filename = '/userdata/smetzger/all_deepul_files/runs/moco_' + custom_aug_name + '.txt'
string = "submit_job -q mind-gpu"
string += " -m 318 -g 4"
string += " -o " + filename
string += ' -n MoCo1'
string += ' -x python /userdata/smetzger/all_deepul_files/deepul_proj/moco/main_moco.py'

# add all the default args: 
string += " -a resnet50 --lr 0.4  --batch-size 512 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1"
string += ' --moco-t 0.2' # MoCov2 arguments. 
string += ' --checkpoint_fp ' + str(checkpoint_fp)
string += ' --rank 0'
string += " --data /userdata/smetzger/data/cifar_10/ --notes 'moco_super'"

string += ' --mlp --cos --epochs 2000'

# Huge line here, submit custom agumentations: 
string += ' --custom_aug_name ' + custom_aug_name
string += ' --checkpoint-interval 250'
cmd = shlex.split(string)
print(cmd)
subprocess.run(cmd, stderr=subprocess.STDOUT)