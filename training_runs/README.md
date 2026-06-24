# training_runs

Fine-tuning (training) runs, one self-contained directory per run: the SLURM
submission script (`train*.sh`, which calls `mattergen-finetune`) next to its stdout
log (`ll_out*`). This mirrors how `examples/li_selective_electrodes/<run>/` organizes the
generation/inference runs (each with its own `run.sh` + outputs + log).

Trained weights are **not** here — `.ckpt` files land in `outputs/singlerun/<date>/` and
are gitignored. The runs below produced the checkpoints later used for generation.

| Run dir | Script | Log | Dataset / notes |
|---|---|---|---|
| `600epochs_LiNa/` | — (script superseded) | `ll_out` | li_data_20, 600 epochs |
| `1500epochs_LiNa_full/` | — (script superseded) | `ll_out_1500epochs` | li_data_20, 1500 epochs, full finetune |
| `1500epochs_LiNa_notfull/` | `train.sh` | `ll_out_1500epochs_notfullfinetune` | li_data_20, not-full finetune |
| `600epochs_LiNadiff/` | `train_diff.sh` | `ll_out_600epochs_diff` | li_data_20_diff (ΔV trick) |
| `600epochs_LiNadiff_add1/` | `train_diff_add1.sh` | `ll_out_600epochs_diff_add1` | li_data_20_diff_add1 (self-training round 1) |
| `600epochs_LiNadiff_add2/` | `train_diff_add2.sh` | `ll_out_600epochs_diff_add2` | li_data_20_diff_add2 (self-training round 2) |
| `li_data_v2/` | `train_li_data_v2.sh` | (not run yet) | new v2 dataset, unfiltered |
| `li_data_v2_20/` | `train_li_data_v2_20.sh` | (not run yet) | v2, ≤20 atoms |
| `li_data_v2_diff/` | `train_li_data_v2_diff.sh` | (not run yet) | v2 + ΔV trick |
| `li_data_v2_20_diff/` | `train_li_data_v2_20_diff.sh` | (not run yet) | v2 ≤20 atoms + ΔV trick |

Submit a run from inside its directory, e.g. `cd training_runs/600epochs_LiNadiff && sbatch train_diff.sh`.
The `#SBATCH --output=` filename is relative to the submission directory, so re-running
writes the log back into the same run dir.
