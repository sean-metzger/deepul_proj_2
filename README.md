# AutoSelf

Example 4-gpu execution/

```bash
CUDA_VISIBLE_DEVICES=0,2,5,7 ./4-gpu-end-to-end.sh $PWD/checkpoints /path/to/cifar-10/ 'Initial test run'
```


To not report a run to wandb do 
```bash
WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=0,2,5,7 ./4-gpu-end-to-end.sh $PWD/checkpoints /path/to/cifar-10/ 'Initial test run'
```
