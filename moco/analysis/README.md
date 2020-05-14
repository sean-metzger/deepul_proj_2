# Analysis Pipeline

## Simple
1. Run the pretraining (e.g. `main-moco.py`) which uploads to wandb
1. Run the evaluation (e.g. `main-lincls.py`) which uploads to wandb 
1. Run `post-analyze.py < id-file.txt` on a file containing the ids of the runs that need a post analysis, if you have one id, simply do `post-analyze.py < 'my-single-id'`, which will run the post analysis provided ids and upload the results directly to wandb. If you need help getting a list of ids, see the next section "Getting a List of Ids".
1. Create a plot by:
    1. Make sure that all needed runs have a post-analysis run on them.
    1. First creating a group of the desired runs on wandb, i.e. in a "report." (this is important so that we can create plots with a direct relationship to a set of wandb runs.)
    1. Export the report to a csv. 
    1. Run the appropriate plot creation script on the exported csv, e.g. `TODO put an example command here`


## Getting a list of ids
1. First creating a group of the desired runs on wandb, i.e. in a "report."
1. Export the report to a csv. 
1. Run the following command on the csv (replace the 2 variables at the start): 

``` shell
WANDBFILE="wandb-output-file.csv"
OUTPUTFILE="output-ids.txt"
awk -F ',' '{print $2}' ${WANDBFILE} | awk -F '"' '{print $2}' | tail -n +2 > ${OUTPUTFILE}
```
