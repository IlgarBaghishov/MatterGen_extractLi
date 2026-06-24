#!/bin/bash

#SBATCH --gres=gpu:a100:1
##SBATCH --gres=shard:a100:6
#SBATCH --job-name=diff
#SBATCH --output=ll_out_600epochs_diff_add1

export PROPERTY1=Li_voltage
export PROPERTY2=Na_voltage
export HYDRA_FULL_ERROR=1

mattergen-finetune adapter.pretrained_name=mattergen_base data_module=li_data_20_diff_add1 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 ~trainer.logger data_module.properties=["$PROPERTY1","$PROPERTY2"]
