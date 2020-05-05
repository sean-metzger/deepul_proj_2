#!/usr/bin/env python

import wandb

if __name__ == "__main__":
    
    wandb.init(project="autoself", id="0DKhLxrlFZ9A78pBjttUxKUoAr8FBnSK", resume=True)
    wandb.config.update({
        "rand_aug": True,
        "rand_aug_m": 5,
        "rand_aug_n": 3,
    })
