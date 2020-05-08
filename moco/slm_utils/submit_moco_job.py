print('Sko Buffs')
import subprocess
import shlex
import os 


checkpoint_fp = '/userdata/smetzger/all_deepul_files/ckpts'
  # elif name == 'rrc_min_max': 
  #       transform_train.transforms.insert(0, Augmentation(fa_rrc_min_max()))
  #   elif name == 'rrc_min_max_weighted': 
  #       transform_train.transforms.insert(0, Augmentation(fa_rrc_min_max_weighted()))
  #   elif name == 'rrc_pure': 
  #       print('not adding anything')
  #   elif name == 'rrc_max_icl_top2': 
  #       transform_train.transforms.insert(0, Augmentation(fa_rrc_max_icl_top2()))
  #   elif name == 'rrc_min_rot_top2': 
  #       transform_train.transforms.insert(0, Augmentation(fa_rrc_min_rot_top2()))
  #   elif name == 'rrc_min_max_top2': 
  #       transform_train.transforms.insert(0, Augmentation(fa_rrc_min_max_top2()))
custom_aug_names = ['augplus']

for custom_aug_name in custom_aug_names: 
 
	filename = '/userdata/smetzger/all_deepul_files/runs/moco_' + custom_aug_name + '_fresh.txt'
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
	string += " --data /userdata/smetzger/data/cifar_10/ --notes 'minmax_fresh_til_2k'"

	string += ' --mlp --cos --epochs 2000'

	# Huge line here, submit custom agumentations: 
	# string += ' --custom_aug_name ' + custom_aug_name

	string += ' --aug-plus'
	string += ' --checkpoint-interval 250'


	# HUGE LINE: only use rand_resize_crop as the base xform.
	# string += ' --rand_resize_only'
	# string += ' --resume ' + checkpoint_fp + '/s7pM4_750epochs_512bsz_0.4000lr_mlp_cos_custom_aug_rrc_min_max_0749.tar'
	# string += ' --start-epoch 749'

	cmd = shlex.split(string)
	print(cmd)
	subprocess.run(cmd, stderr=subprocess.STDOUT)