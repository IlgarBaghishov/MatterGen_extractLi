# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Relaxation engine for post-processing/evaluation.
# The original MatterGen implementation relaxed structures with MatterSim.
# Here the engine is replaced by the UMA universal MLIP (FAIRChem), which is the
# same potential used for the Li/Na voltage post-processing, so that the metrics
# path (energy-above-hull, etc.) and the voltage path share a single relaxer.

import sys

import numpy as np
import torch
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import FIRE
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# --- UMA / relaxation defaults (kept in sync with mattergen/scripts/post_process.py) ---
UMA_MODEL_NAME = "uma-m-1p1"
UMA_TASK_NAME = "omat"
RELAX_FMAX = 0.05
RELAX_MAX_STEPS = 2000

# Lazily-initialised module-level singleton so the model is loaded once per process.
_UMA_CALCULATOR = None


def get_device(device: str | None = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_uma_calculator(
    model_name: str = UMA_MODEL_NAME,
    task_name: str = UMA_TASK_NAME,
    device: str | None = None,
):
    """Initialise (once) and return the FAIRChem UMA calculator."""
    global _UMA_CALCULATOR
    if _UMA_CALCULATOR is None:
        try:
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
        except ImportError as e:
            raise ImportError(
                "fairchem-core is required for UMA relaxation. Install fairchem-core."
            ) from e
        device = get_device(device)
        print(f"Loading UMA potential: {model_name} (task={task_name}) on {device}...")
        predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
        _UMA_CALCULATOR = FAIRChemCalculator(predictor, task_name=task_name)
    return _UMA_CALCULATOR


def relax_single_atoms(
    atoms: Atoms,
    calc=None,
    fmax: float = RELAX_FMAX,
    steps: int = RELAX_MAX_STEPS,
    logfile=None,
) -> tuple[Atoms, float, bool]:
    """Relax one ASE Atoms with UMA (FIRE + FrechetCellFilter, i.e. cell + positions).

    Returns (relaxed_atoms, total_potential_energy, converged). The input ``atoms``
    is not mutated (a copy is relaxed).
    """
    if calc is None:
        calc = get_uma_calculator()
    atoms = atoms.copy()
    atoms.calc = calc
    dyn = FIRE(FrechetCellFilter(atoms), logfile=(sys.stdout if logfile is None else logfile))
    converged = dyn.run(fmax=fmax, steps=steps)
    energy = atoms.get_potential_energy()
    return atoms, energy, converged


def relax_atoms(
    atoms: list[Atoms],
    device: str = "cuda",
    potential_load_path: str | None = None,
    output_path: str | None = None,
    calc=None,
    fmax: float = RELAX_FMAX,
    steps: int = RELAX_MAX_STEPS,
    **kwargs,
) -> tuple[list[Atoms], np.ndarray]:
    """Relax a list of ASE Atoms with UMA.

    ``potential_load_path`` is accepted for backward compatibility with the
    previous MatterSim-based signature but is ignored; the UMA model is selected
    by ``UMA_MODEL_NAME``. Returns (relaxed_atoms, total_energies).
    """
    if calc is None:
        calc = get_uma_calculator(device=device)
    relaxed_atoms: list[Atoms] = []
    total_energies: list[float] = []
    for a in atoms:
        ra, energy, _ = relax_single_atoms(a, calc=calc, fmax=fmax, steps=steps)
        relaxed_atoms.append(ra)
        total_energies.append(energy)
    total_energies = np.array(total_energies)
    if output_path:
        write(output_path, relaxed_atoms, format="extxyz")
    return relaxed_atoms, total_energies


def relax_structures(
    structures: Structure | list[Structure],
    device: str = "cuda",
    potential_load_path: str | None = None,
    output_path: str | None = None,
    **kwargs,
) -> tuple[list[Structure], np.ndarray]:
    if isinstance(structures, Structure):
        structures = [structures]
    atoms = [AseAtomsAdaptor.get_atoms(s) for s in structures]
    relaxed_atoms, total_energies = relax_atoms(
        atoms,
        device=device,
        potential_load_path=potential_load_path,
        output_path=output_path,
        **kwargs,
    )
    relaxed_structures = [AseAtomsAdaptor.get_structure(a) for a in relaxed_atoms]
    return relaxed_structures, total_energies
