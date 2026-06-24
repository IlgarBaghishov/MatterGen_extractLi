#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=li_data_v2_20_diff
#SBATCH --output=ll_out_li_data_v2_20_diff

export PROPERTY1=Li_voltage
export PROPERTY2=Na_voltage
export HYDRA_FULL_ERROR=1

# NOTE: for *_diff datasets, the Na_voltage column holds ΔV = Li_voltage - Na_voltage (see CLAUDE.md).
mattergen-finetune adapter.pretrained_name=mattergen_base data_module=li_data_v2_20_diff +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 ~trainer.logger data_module.properties=["$PROPERTY1","$PROPERTY2"]
