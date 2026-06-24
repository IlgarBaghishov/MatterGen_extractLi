# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository. It records what
this project is, how the pipeline fits together, where each piece lives, and the
non-obvious conventions — most importantly the **ΔV (voltage-difference) trick**.

## What this repository is

`MatterGen_extractLi` is a closed-loop generative workflow for discovering
**lithium-selective intercalation electrodes** (for electrochemical Li extraction
from brine, where Na⁺ competes with Li⁺). A property-conditioned diffusion model
(MatterGen) is fine-tuned on Li and Na intercalation voltages, conditioned on the
**Li–Na voltage difference ΔV** to break the natural Li/Na voltage correlation,
iteratively retrained on its own high-selectivity outputs, and the generated
structures are screened with the **UMA universal MLIP** (FAIRChem).

Associated paper: "Discovery of Lithium-Selective Electrodes with Generative AI"
(Baghishov, Jung, Liu, Henkelman; UT Austin). The completed pipeline yields a small
set of dynamically stable, Li-selective candidate electrodes.

### Built on top of (acknowledge, don't re-derive)
- **Microsoft MatterGen** (https://github.com/microsoft/mattergen, MIT) — this repo is
  a fork. The upstream README is preserved verbatim as `README_MATTERGEN.md`.
- **Meta FAIRChem / UMA** (https://github.com/FAIR-Chem/fairchem) — universal MLIP used
  for all relaxations and energies in post-processing.
- **MPElectroML** (https://github.com/IlgarBaghishov/MPElectroML) — the **external**
  package that builds the conditioning dataset from the Materials Project + UMA. It is
  NOT included here; its output CSVs are the training data under `datasets/`.

## The end-to-end pipeline (where each stage lives)

```
Materials Project + UMA ──(external: MPElectroML)──► absolute Li_voltage & Na_voltage CSVs
        │                                                     (datasets/li_data_20*/{train,val,test}.csv)
        │   ΔV TRICK: overwrite Na_voltage column with (Li_voltage − Na_voltage)   ← see below
        ▼
   datasets/li_data_20_diff*  ──(csv-to-dataset)──►  datasets/cache/<name>/{train,val,test}/*.npy
        │
        ▼
   mattergen-finetune  (conditions on Li_voltage + Na_voltage; with _diff data, "Na_voltage" IS ΔV)
        │                       train*.sh  →  outputs/singlerun/<date>/.../*.ckpt   (weights, gitignored)
        ▼
   mattergen-generate  --model_path=outputs/... --properties_to_condition_on="{'Li_voltage':…,'Na_voltage':…}"
        │                       → generated_crystals_cif.zip / .extxyz / generated_trajectories.zip
        ▼
   mattergen-postprocess (UMA): relax host / Li / Na forms → voltages → stability/novelty metrics
        │                       → voltage_analysis.csv  +  evaluation_metrics.json
        ▼
   screening + combine (examples/li_selective_electrodes/combine_upto_incl_add2/*.ipynb)
                                → merged_data.csv, final_filtered_data.csv, figures
```

## ⚠️ The ΔV trick (most important non-obvious thing)

The paper "conditions on the voltage difference ΔV = V_Li − V_Na", but **no model code
knows about ΔV**. It is realized purely as a dataset column transform:

- In the plain datasets (`li_data_20`, etc.) the columns are the **absolute** voltages:
  `Li_voltage = V_Li`, `Na_voltage = V_Na` (both vs SHE).
- In the **`_diff` datasets** (`li_data_20_diff`, `_diff_add1`, `_diff_add2`):
  - `Li_voltage` is **unchanged** (still V_Li), but
  - **`Na_voltage` is OVERWRITTEN with `Li_voltage − Na_voltage` (= ΔV).**

  Verified example: a row with `V_Li=-1.553, V_Na=-1.274` becomes
  `Li_voltage=-1.553, Na_voltage=-0.279` (because `-1.553 − (-1.274) = -0.279`).

Consequence: a model trained on a `_diff` dataset that "conditions on `Na_voltage`" is
really conditioning on **ΔV**. That is why `_diff` generation runs use targets like
`--properties_to_condition_on="{'Li_voltage': 0.2, 'Na_voltage': 2.0}"` — meaning
"V_Li ≈ 0.2 V (inside the water window) AND ΔV ≈ 2.0 V (strongly Li-selective)".

Where the trick is applied: it is a **separate post-step**, NOT part of MPElectroML or
the `examples` preprocessing (those produce only absolute voltages). Historically it
lived in a dataset-prep notebook (`li_extraction/plots.ipynb`). It is a trivial column
transform — for each split CSV in a dataset folder:

    df["Na_voltage"] = df["Li_voltage"] - df["Na_voltage"]   # keep Li_voltage unchanged

(loop over train/val/test, write to `<name>_diff/`). After creating a `_diff` folder,
build its cache with `csv-to-dataset` and add a `data_module/<name>_diff.yaml` config
before training.

## Datasets and naming conventions

CSVs live in `datasets/<name>/{train,val,test}.csv`; the cache that training reads is
`datasets/cache/<name>/{train,val,test}/*.npy + Li_voltage.json + Na_voltage.json`,
built by `csv-to-dataset`. The `data_module` config `mattergen/conf/data_module/<name>.yaml`
points at `${PROJECT_ROOT}/../datasets/cache/<name>`.

CSV columns: `pretty_formula, material_id, num_atoms, host_energy_per_atom,
Li_energy_per_atom, Na_energy_per_atom, Li_voltage, Na_voltage, cif`
(training only needs `cif` + `Li_voltage` + `Na_voltage` + an id).

Name conventions:
- `li_data_20` — Li electrode data filtered to **≤ 20 atoms/cell** ("20" = the filter).
- `li_data_all` / unfiltered variants — all cell sizes.
- `_diff` — the ΔV trick applied (Na_voltage = V_Li − V_Na).
- `_add1`, `_add2` — **iterative self-training** augmentations: high-selectivity generated
  structures from round N are added back into the training set for round N+1.
- `li_data_v2`, `li_data_v2_20` — NEW datasets (from William/J. Liu): `li_data_v2` =
  unfiltered, `li_data_v2_20` = filtered to ≤20 atoms. (To train `_diff`-style models on
  these, first build `li_data_v2_diff` / `li_data_v2_20_diff` via the ΔV trick above.)

## Intercalation voltage definition (for post-processing)

Voltage vs SHE for inserting n ions of charge z (computed from UMA relaxed total energies):

  V_X = −(E_{X_n·host} − E_host − n·μ_X) / (n·z) + E°(X⁺/X)

with UMA bulk references `μ_Li = −1.9032`, `μ_Na = −1.3093` eV/atom and SHE shifts
`E°(Li) = −3.04 V`, `E°(Na) = −2.71 V`. Water-stability window ≈ −0.41 to +0.82 V; a host
is Li-selective when V_Li is in-window and ΔV = V_Li − V_Na is large/positive. These
constants live at the top of `mattergen/scripts/post_process.py`
(`BULK_ENERGIES`, `VALENCIES`, `SHE_SHIFT`).

## Post-processing: UMA replaces MatterSim (key fork change)

Upstream MatterGen relaxes with MatterSim. Here the relaxation engine is **UMA**
(`uma-m-1p1`, `omat` task, ASE FIRE + FrechetCellFilter), used as the single shared
engine for both the voltage analysis and the metrics:

- `mattergen/evaluation/utils/relaxation.py` — the UMA engine: `get_uma_calculator()`
  (lazy FAIRChem singleton), `relax_single_atoms()`, `relax_atoms()`, `relax_structures()`.
  (Upstream MatterSim code was removed; the old stub that did nothing is gone.)
- `mattergen/scripts/post_process.py` — per-structure logic: relax the Li form, swap
  Li→Na and relax, strip the ion to get the host and relax, then compute V_Li and V_Na.
  It delegates relaxation to the shared engine above.
- `mattergen/scripts/batch_post_process.py` — batch driver over a folder of CIFs; writes
  `voltage_analysis.csv`, then runs `evaluate()` for stability/uniqueness/novelty vs the
  MP2020 reference. Exposed as the `mattergen-postprocess` console command.
- `mattergen/evaluation/` is otherwise the MatterGen evaluation module with the user's
  modifications; energy-above-hull is computed from UMA energies + the MP2020 correction
  against the reference dataset.

Important: post-processing is **integrated into the package** — never copy these scripts
into a results folder and run them there (that was the old workflow). Run
`mattergen-postprocess` against any CIF directory instead.

## The MP2020 reference dataset and Git LFS (non-obvious)

`data-release/alex-mp/reference_MP2020correction.gz` (~833 MB, 845,997 MP+Alexandria
entries) is required at runtime by the metrics (`mattergen/evaluation/reference/presets.py`
loads it). It is **byte-identical to MatterGen's** (same LFS oid `c722f72c…`), so:

- `.lfsconfig` points Git LFS at **`https://github.com/microsoft/mattergen.git/info/lfs`**
  (`auth=none`, verified servable). This repo stores only the LFS **pointer** and does NOT
  consume its own LFS quota.
- `fetchexclude = *`, so a fresh clone gets the pointer only. Fetch the real file with:
  `git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""`
- On push, if Git LFS tries to upload it (it already exists upstream), use
  `git push --no-verify` if needed; clones still resolve it from Microsoft.

The unused Microsoft pretrained-checkpoint LFS pointers were removed (the `checkpoints/`
dir is gone); the MatterGen **base** model is downloaded from Hugging Face on demand.

## Installation (VERIFIED recipe — the naive one does NOT work)

Requires Python 3.10, a CUDA GPU, and `git lfs`. Installing FAIRChem upgrades torch
(2.2.1+cu118 → 2.8.0+cu128), which breaks MatterGen's pinned PyG/torchvision/numpy, so
those must be rebuilt. The following is **tested working** (MatterGen generation + UMA
relaxation, both on an A100), and is the env `mattergenLi`:

```bash
conda create -n mattergenLi python=3.10 -y && conda activate mattergenLi
pip install uv && uv pip install -e .                 # MatterGen (torch 2.2)
pip install fairchem-core                              # UMA (bumps torch → 2.8.0+cu128)
pip install --force-reinstall --no-cache-dir torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html  # rebuild PyG for torch 2.8
pip install torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "numpy<2"                                  # gemnet uses np.math (removed in numpy 2)
```

How they coexist: there is exactly **one** torch (`2.8.0+cu128`) and one numpy
(`1.26.4`). MatterGen runs on a newer torch than it declares (its pins are advisory; the
code uses only generic ops), FAIRChem runs on an older numpy than it declares. The
version-sensitive **compiled extensions** (torch_scatter/sparse/cluster, torchvision,
torchaudio) are all rebuilt for torch 2.8 so their ABI matches the single torch. The pip
"incompatible" warnings about MatterGen's declared pins are expected and harmless.

GPU check: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
→ `2.8.0+cu128 True`.

## What is gitignored (and what a fresh clone is missing)

Rule: ignore anything > 50 MB **except** the LFS reference. Patterns in `.gitignore`:
`outputs/**/*.ckpt` (trained weights, ~13 GB) and `**/generated_trajectories.zip`
(~20 GB). Also `*.egg-info/` (editable-install metadata).

Therefore a fresh `git clone` HAS: all code, configs, `train*.sh`, the training CSVs AND
their preprocessed cache, run provenance (hydra configs/hparams/metrics.csv per run),
results (notebooks, `voltage_analysis.csv`, generated CIF zips, final screening CSVs/
figures), and the LFS reference pointer. It does NOT have: the trained `.ckpt` weights or
the full denoising trajectories. To get trained models on another machine, either
**re-train** (data+configs are present) or copy `outputs/*.ckpt` out-of-band.

## Repo layout

- `mattergen/` — the Python package (importable as `mattergen`). Diffusion model,
  `conf/` (incl. `Li_voltage`/`Na_voltage` property embeddings, `data_module/li_data_20*`,
  `finetune.yaml`), `evaluation/` (UMA post-processing), `scripts/`
  (`finetune`, `generate`, `run`, `csv_to_dataset`, `post_process`, `batch_post_process`).
- `datasets/` — training CSVs (`li_data_20*`, `li_data_v2*`) + `cache/` (what training reads).
- `data-release/alex-mp/reference_MP2020correction.gz` — MP2020 reference (LFS, from Microsoft).
- `outputs/singlerun/<date>/` — training runs (configs/logs tracked; `.ckpt` gitignored).
- `examples/li_selective_electrodes/` — ΔV-conditioned generation runs (`600epochs_LiNadiff`,
  `_add1`, `_add2`) and the final screening in `combine_upto_incl_add2/`.
- `train*.sh` — SLURM training drivers (`mattergen-finetune` with Li/Na voltage embeddings).
- `README.md` (this repo) / `README_MATTERGEN.md` (upstream, preserved).

## Command cheat sheet

```bash
# Build a dataset cache from CSVs
csv-to-dataset --csv-folder datasets/<name> --dataset-name <name> --cache-folder datasets/cache/<name>

# Fine-tune (base model auto-downloaded from HF); see train_diff_add2.sh
export PROPERTY1=Li_voltage PROPERTY2=Na_voltage
mattergen-finetune adapter.pretrained_name=mattergen_base data_module=<name> \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \
  +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \
  ~trainer.logger data_module.properties=["$PROPERTY1","$PROPERTY2"]

# Generate (with a _diff model, 'Na_voltage' here is ΔV)
mattergen-generate <out> --model_path=outputs/singlerun/<run> --batch_size=256 --num_batches=8 \
  --properties_to_condition_on="{'Li_voltage': 0.2, 'Na_voltage': 2.0}" --diffusion_guidance_factor=2.0

# Post-process with UMA (fetch reference once, then run)
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""
unzip <out>/generated_crystals_cif.zip -d <out>/generated_crystals_cifs
mattergen-postprocess --directory_path=<out>/generated_crystals_cifs   # add --relax to re-relax with UMA
```

## Gotchas / reminders

- `material_id` is NOT unique across rows — joins on it can multiply rows; dedupe carefully.
- Always run `mattergen-postprocess` from the installed package; do not copy scripts into result dirs.
- The `_diff` ΔV trick must be re-applied when creating any new `_diff` dataset.
- MD-stability (annealing 300→800 K) is done off-repo (Sung Hoon Jung's machine), not here.
- GenLiIntercal (the old Pawan-era superconductor repo) is unrelated; its only relevance was
  that the post-processing scripts originated there before being integrated into this package.
