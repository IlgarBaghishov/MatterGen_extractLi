#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=02_2_124
#SBATCH --output=ll_out

mattergen-generate . --model_path=/home/ilgar/MatterGen_extractLi/outputs/singlerun/2025-12-08/19-42-57  --batch_size=100 --properties_to_condition_on="{'Li_voltage': 0.2, 'Na_voltage': 2.0}" --diffusion_guidance_factor=4.0 --num_batches=1

unzip generated_crystals_cif.zip -d generated_crystals_cifs
cd generated_crystals_cifs

source $HOME/code/miniforge3/etc/profile.d/conda.sh
conda activate eonmeta

mattergen-postprocess --directory_path="."
