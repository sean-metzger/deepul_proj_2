# Things on wandb got a little messed up
# For certain numbers, I placed the command I used to find them in the wandb directory on millennium machines 

iresults = {
    #####################
    # RAND AUG 15 EVALS #
    #####################
    "lYYsDrHaRXSGZe6WNXHFEBaKWVPYS3IZ": {
        "description": "randaug n=2 m=11",
        "epochs": 100,
        "val-rotate": 71.531,
        "val-classify": 62.602,
        "type": "randaug"
    },
    "lYYsDrHaRXSGZe6WNXHFEBaKWVPYS3IZ_7X7YL": {
        "description": "randaug n=2 m=11 at 60 epochs (rotation is on 100 run on wandb)",
        "epochs": 60,
        "val-rotate": 70.764, # find . -type d -name '*lYYsDrHaRXSGZe6WNXHFEBaKWVPYS3IZ' | awk '{ print $1"/wandb-history.jsonl" }' | xargs grep 60epoch-val | cut -d ':' -f 1 --complement | jq '.["60epoch-val-rotation"]' | sort

        "val-classify": 59.480, # find . -type d -name '*lYYsDrHaRXSGZe6WNXHFEBaKWVPYS3IZ*' | awk '{ print $1"/wandb-history.jsonl" }' | xargs grep 60epoch-val | cut -d ':' -f 1 --complement | jq '.["60epoch-val-classify"]' | sort
        "type": "randaug"
    },

    "lYYsDrHaRXSGZe6WNXHFEBaKWVPYS3IZ_7X7YL": {
        "description": "randaug n=2 m=11 at 20 epochs",
        "epochs": 20,
        "val-rotate": -1,
        "val-classify": -1,
        "type": "randaug"
    },


    "PoKW7CTv2cedBdUFDxKkNaqxKuT7iBWc": {
        "description": "randaug n=2 m=7",
        "epochs": 100,
        "val-rotate": 72.213, # find . -type d -name '*PoKW7CTv2cedBdUFDxKkNaqxKuT7iBWc' | awk '{ print $1"/wandb-history.jsonl" }' | xargs grep '"val-rotat' | cut -d ':' -f 1 --complement | jq '.["val-rotation"]' | sort
        "val-classify": 62.462, #  find . -type d -name '*PoKW7CTv2cedBdUFDxKkNaqxKuT7iBWc' | awk '{ print $1"/wandb-history.jsonl" }' | xargs grep '"val-classify' | cut -d ':' -f 1 --complement | jq '.["val-classify"]' | sort
        "type": "randaug"
    },
    "PoKW7CTv2cedBdUFDxKkNaqxKuT7iBWc_60": {
        "description": "randaug n=2 m=7 at 60 epochs",
        "epochs": 60,
        "val-rotate": 70.758, # find . -type d -name '*PoKW7CTv2cedBdUFDxKkNaqxKuT7iBWc' | awk '{ print $1"/wandb-history.jsonl" }' | xargs grep 60epoch-val | cut -d ':' -f 1 --complement | jq '.["60epoch-val-rotation"]'
        "val-classify": 59.586, # find . -type d -name '*PoKW7CTv2cedBdUFDxKkNaqxKuT7iBWc' | awk '{ print $1"/wandb-history.jsonl" }' | xargs grep 60epoch-val | cut -d ':' -f 1 --complement | jq '.["60epoch-val-classify"]' | sort
        "type": "randaug"
    },
    "PoKW7CTv2cedBdUFDxKkNaqxKuT7iBWc_20": {
        "description": "randaug n=2 m=7 at 20 epochs",
        "epochs": 20,
        "val-rotate": -1,
        "val-classify": -1,
        "type": "randaug"
    },



    "TiSj6QpyCyjM4bOeJJoctmc1VQXBBSZ6": {
        "description": "randaug n=2 m=5",
        "epochs": 100,
        "val-rotate": 71.426, # TODO was this done wrong? find . -type d -name '*TiSj6QpyCyjM4bOeJJoctmc1VQXBBSZ6*' | awk '{ print $1"/wandb-history.jsonl" }' | xargs grep '"val-rotation' | cut -d ':' -f 1 --complement | jq '.["val-rotation"]' | sort
        "val-classify": 62.446, # find . -type d -name '*TiSj6QpyCyjM4bOeJJoctmc1VQXBBSZ6*' | awk '{ print $1"/wandb-history.jsonl" }' | xargs grep '"val-classify' | cut -d ':' -f 1 --complement | jq '.["val-classify"]' | sort
        "type": "randaug"
    },
    "TiSj6QpyCyjM4bOeJJoctmc1VQXBBSZ6_60": {
        "description": "randaug n=2 m=5 at 60 epochs",
        "epochs": 60,
        "val-rotate": 71.328,
        "val-classify": 60.230,
        "type": "randaug"
    },
    "TiSj6QpyCyjM4bOeJJoctmc1VQXBBSZ6_20": {
        "description": "randaug n=2 m=5 at 20 epochs",
        "epochs": 20,
        "val-rotate": -1,
        "val-classify": -1,
        "type": "randaug"
    },

    "HvOSNB0Nto3NJh0CFiXp8Q3ShCXRj4A4": {
        "description": "randaug n=2 m=9",
        "epochs": 100,
        "val-rotate": 73.217,
        "val-classify": 64.074,
        "type": "randaug"
    },
    "HvOSNB0Nto3NJh0CFiXp8Q3ShCXRj4A4_60": {
        "description": "randaug n=2 m=9 at 60 epochs",
        "epochs": 60,
        "val-rotate": 71.403,
        "val-classify": 60.752, # TODO why was this such a big drop
        "type": "randaug"
    },
    "HvOSNB0Nto3NJh0CFiXp8Q3ShCXRj4A4_20": {
        "description": "randaug n=2 m=9 at 20 epochs",
        "epochs": 20,
        "val-rotate": -1,
        "val-classify": -1,
        "type": "randaug"
    },

    "dNjCJlYzUVPBN1KigDDp31tHm0i41SIo": {
        "description": "randaug n=2 m=13",
        "epochs": 100,
        "val-rotate": 70.625,
        "val-classify": 61.716,
        "type": "randaug"
    },
    "dNjCJlYzUVPBN1KigDDp31tHm0i41SIo_60": {
        "description": "randaug n=2 m=13 at 60 epochs",
        "epochs": 60,
        "val-rotate": 70.268,
        "val-classify": 59.926,
        "type": "randaug"
    },
    "dNjCJlYzUVPBN1KigDDp31tHm0i41SIo_20": {
        "description": "randaug n=2 m=13 at 20 epochs",
        "epochs": 20,
        "val-rotate": -1,
        "val-classify": -1,
        "type": "randaug"
    },
    
    
}
