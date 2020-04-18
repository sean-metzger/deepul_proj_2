# AutoSelf

Example 4-gpu execution/

```bash
CUDA_VISIBLE_DEVICES=0,2,5,7 ./4-gpu-end-to-end.sh $PWD/checkpoints/ /path/to/cifar-10/ 'Initial test run'
```


To not report a run to wandb do 
```bash
WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=0,2,5,7 ./4-gpu-end-to-end.sh $PWD/checkpoints/ /path/to/cifar-10/ 'Initial test run'
```


# Open questions:
* What learning rate and temperature should we use for moco?
* What we use the cosine annealing learning scheduler?
* Should our other project heads all have an MLP?
* How can we simulataneously measure the linear classifier performance throughout training? i.e. how should we merge the two scripts?


# TODO:
* Save the final/best models to wandb for each run.
