# MatterGen_extractLi

**Generative discovery of lithium-selective intercalation electrodes for electrochemical Li extraction.**

This repository implements a closed-loop generative workflow: a property-conditioned
diffusion model (MatterGen) is fine-tuned on Li and Na intercalation voltages, conditioned
on the **Li–Na voltage difference** (ΔV) to break the natural Li/Na voltage correlation,
iteratively retrained on its own high-selectivity outputs, and the generated structures are
screened with a universal machine-learned interatomic potential (**UMA**, via FAIRChem).

> **This repo is built on top of [Microsoft MatterGen](https://github.com/microsoft/mattergen)
> and [Meta FAIRChem](https://github.com/FAIR-Chem/fairchem).** Most of the code is MatterGen
> (MIT-licensed); the upstream documentation is preserved in
> [`README_MATTERGEN.md`](./README_MATTERGEN.md). See [Relationship to MatterGen](#relationship-to-mattergen)
> for exactly what was changed.

## Table of contents
- [Relationship to MatterGen](#relationship-to-mattergen)
- [Installation](#installation)
- [Git LFS: the reference dataset (fetched from Microsoft)](#git-lfs-the-reference-dataset-fetched-from-microsoft)
- [Usage](#usage)
- [What is gitignored](#what-is-gitignored)
- [Acknowledgments & citation](#acknowledgments--citation)

## Relationship to MatterGen

This project is a fork of MatterGen with a small number of targeted changes:

- **Custom property conditioning** — added `Li_voltage` and `Na_voltage` as conditioning
  properties (`mattergen/conf/.../property_embeddings/`, registered in
  `mattergen/common/utils/globals.py`), and a finetune config defaulting to the Li dataset.
- **Post-processing engine: MatterSim → UMA.** MatterGen's evaluation relaxes structures
  with MatterSim. Here the relaxer is replaced by the **UMA universal MLIP** (FAIRChem,
  `uma-m-1p1`, FIRE + FrechetCellFilter) in `mattergen/evaluation/utils/relaxation.py`, used
  as the single shared engine for both metrics and voltage analysis.
- **Integrated voltage post-processing** — `mattergen/scripts/post_process.py` and
  `batch_post_process.py`, exposed as the `mattergen-postprocess` command, compute Li and Na
  intercalation voltages (host / Li-inserted / Na-inserted relaxations) and then run the
  stability/uniqueness/novelty metrics. No more copying scripts into result folders.
- **Data & results** — `datasets/li_data_20*` (training CSVs) and
  `examples/li_selective_electrodes/` (ΔV-conditioned generation runs + final screening).
- **Reference dataset via Microsoft's LFS** — `data-release/alex-mp/reference_MP2020correction.gz`
  is byte-identical to MatterGen's, so it is fetched from Microsoft's public Git LFS instead of
  being re-hosted here (see below).

## Installation

Requires **Python 3.10**, a **CUDA GPU**, and **Git LFS**. The MatterGen install steps below
are the same as upstream; the only addition is FAIRChem for the UMA post-processing.

### 1. Git LFS (same as MatterGen)
```bash
git lfs --version            # if missing:
sudo apt install git-lfs
git lfs install
```

### 2. Python environment + MatterGen (same as MatterGen)
MatterGen pins specific CUDA wheels (`torch==2.2.1+cu118`, PyG wheels) via `uv` sources, so
install with [`uv`](https://docs.astral.sh/uv/) as upstream recommends:
```bash
# uv venv (MatterGen-native)
pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```
or into a conda env:
```bash
conda create -n mattergenLi python=3.10 -y
conda activate mattergenLi
pip install uv
uv pip install -e .
```

### 3. FAIRChem / UMA for post-processing (new vs MatterGen)
Post-processing uses the UMA universal MLIP instead of MatterSim, so FAIRChem must be
installed **in the same environment**. Installing it upgrades PyTorch
(`2.2.1+cu118` → `2.8.0+cu128`), which then requires rebuilding the PyG/torchvision wheels
for the new torch and pinning numpy back below 2. The following sequence is **tested working**
(MatterGen generation + UMA post-processing, both on an NVIDIA A100):
```bash
# 3a. UMA / FAIRChem (this upgrades torch to 2.8.0+cu128)
pip install fairchem-core

# 3b. rebuild the PyG extensions for the new torch
pip install --force-reinstall --no-cache-dir torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# 3c. matching torchvision/torchaudio for torch 2.8 (cu128)
pip install torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 3d. MatterGen's gemnet uses np.math (removed in numpy 2); FAIRChem still runs under numpy<2
pip install "numpy<2"
```
> `pip` will print "incompatible" warnings about MatterGen's *declared* pins
> (`torch==2.2.1+cu118`, etc.). These are expected and harmless — the code runs on torch 2.8.
>
> The first UMA call downloads the `uma-m-1p1` weights from Hugging Face; you may need
> `huggingface-cli login` and to accept the model license once.

### 4. Verify the GPU is visible
```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# expect: torch 2.8.0+cu128 cuda True
```
Both MatterGen (generation/training) and UMA (post-processing) use the GPU. A quick end-to-end
check:
```bash
# MatterGen generation (downloads mattergen_base from HF, writes generated_crystals_cif.zip)
mattergen-generate /tmp/gen_test --pretrained-name=mattergen_base --batch_size=1 --num_batches=1
```

## Git LFS: the reference dataset (fetched from Microsoft)

The energy-above-hull / novelty metrics need
`data-release/alex-mp/reference_MP2020correction.gz` (845,997 MP + Alexandria entries with
MP2020 corrections). This repo does **not** store the blob in its own LFS — `.lfsconfig` points
Git LFS at `microsoft/mattergen`, so the identical file is fetched from Microsoft's public LFS:
```bash
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""
```
(Same command as MatterGen's README; it just resolves the object from Microsoft.)
`mattergen/evaluation/reference/presets.py` loads it from this path automatically.

## Usage

### Fine-tune on Li/Na voltages
The base MatterGen checkpoint is downloaded from Hugging Face automatically. See
`training_runs/600epochs_LiNadiff_add2/train_diff_add2.sh`; the core command is:
```bash
export PROPERTY1=Li_voltage PROPERTY2=Na_voltage
mattergen-finetune \
  adapter.pretrained_name=mattergen_base \
  data_module=li_data_20_diff_add2 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \
  ~trainer.logger \
  data_module.properties=["$PROPERTY1","$PROPERTY2"]
```
Training CSVs live in `datasets/li_data_20*`. To build a dataset from your own CSV:
```bash
csv-to-dataset --csv-folder <path> --dataset-name <name> --cache-folder datasets/cache/<name>
```

### Generate (ΔV-conditioned)
```bash
mattergen-generate <out_dir> \
  --model_path=outputs/singlerun/<run> \
  --batch_size=256 --num_batches=8 \
  --properties_to_condition_on="{'Li_voltage': 0.2, 'Na_voltage': 2.0}" \
  --diffusion_guidance_factor=2.0
```

### Post-process with UMA (compute voltages + metrics)
```bash
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""   # once
unzip <out_dir>/generated_crystals_cif.zip -d <out_dir>/generated_crystals_cifs
mattergen-postprocess --directory_path=<out_dir>/generated_crystals_cifs
```
- Relaxes each structure's host / Li-inserted / Na-inserted forms with UMA and writes
  `voltage_analysis.csv` (Li and Na intercalation voltages on the SHE scale).
- Then computes stability (energy above the MP2020 hull), uniqueness, and novelty.
- `--relax` re-relaxes the structures with UMA inside the metrics step.
- Ion settings (target/replacement ion, bulk reference energies, valencies, SHE shifts) are at
  the top of `mattergen/scripts/post_process.py`.

### Results
`examples/li_selective_electrodes/` contains the ΔV-conditioned generation runs (`600epochs_LiNadiff`,
`…_add1`, `…_add2`) and the final merged screening in `combine_upto_incl_add2/`
(merged data, figures, and the final Li-selective candidate set).

## What is gitignored

Anything over 50 MB except the LFS reference: fine-tuned checkpoints (`outputs/**/*.ckpt`) and
sampling trajectory archives (`generated_trajectories.zip`). These stay on local disk but are not
tracked. Run configs/logs under `outputs/` are kept as provenance.

## Acknowledgments & citation

This work is built on top of two projects — please cite them if you use this repository:

- **MatterGen** — Zeni, C., Pinsler, R., Zügner, D. et al. *A generative model for inorganic
  materials design.* Nature 639, 624–632 (2025). https://github.com/microsoft/mattergen (MIT License).
- **FAIRChem / UMA** — the Universal Models for Atoms (UMA) universal MLIP.
  https://github.com/FAIR-Chem/fairchem.

The MP2020-correction reference dataset is hosted by `microsoft/mattergen` via Git LFS.
Upstream MatterGen documentation is preserved in [`README_MATTERGEN.md`](./README_MATTERGEN.md).
