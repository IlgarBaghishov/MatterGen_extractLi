#!/usr/bin/env python
# -*-
"""
Refactored post_process.py with dynamic ion selection and user-configurable constants.
"""

import sys
import warnings
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
import torch

# Shared UMA relaxer (single engine for both voltage post-processing and metrics).
from mattergen.evaluation.utils.relaxation import get_uma_calculator, relax_single_atoms

# =============================================================================
#   USER CONFIGURATION
# =============================================================================

# 1. Define your Ions
TARGET_ION = "Li"         # The ion currently in your input CIF (e.g., Li)
REPLACEMENT_ION = "Na"    # The ion you want to swap in (e.g., Na)

# 2. Reference Energies (eV per atom) for the bulk metal
#    (Update these values based on your specific MLIP potential outputs)
BULK_ENERGIES = {
    #"Li": -2.377,
    #"Na": -3.548,
    #"Mg": -1.600,
    "Li": -1.9032,
    "Na": -1.3093,
    "Mg": -1.51105,
}

# 3. Ion Valencies (Charge 'z')
#    Required to convert eV/atom energy difference into Voltage (V)
VALENCIES = {
    "Li": 1.0,
    "Na": 1.0,
    "K":  1.0,
    "Mg": 2.0,
    "Ca": 2.0,
    "Al": 3.0
}

SHE_SHIFT = {
    "Li": -3.04,
    "Na": -2.71, 
    "K": -2.92,
    "Mg": -2.37,
    "Ca": -2.87,
    "Al": -1.66,
}


# 4. Model Settings
UMA_MODEL_NAME = "uma-m-1p1"
UMA_TASK_NAME = "omat"
RELAX_FMAX = 0.05
RELAX_MAX_STEPS = 2000
OUT_DIR = "postprocess_output"
NUM_THREADS = 8

# =============================================================================

torch.set_num_threads(NUM_THREADS)

# --- fairchem.core Imports ---
try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
except ImportError:
    pass

def get_ml_potential() -> 'FAIRChemCalculator':
    """Initializes and returns the FAIRChem UMA calculator (shared singleton)."""
    return get_uma_calculator(model_name=UMA_MODEL_NAME, task_name=UMA_TASK_NAME)

def get_relaxed_energy(structure: Structure, calc: 'FAIRChemCalculator') -> float:
    """Relaxes a structure using ASE FIRE optimizer and returns potential energy."""
    try:
        atoms = AseAtomsAdaptor.get_atoms(structure)
        # Shared UMA relaxer (FIRE + FrechetCellFilter); logs to stdout so it's captured.
        relaxed_atoms, final_energy, converged = relax_single_atoms(
            atoms, calc=calc, fmax=RELAX_FMAX, steps=RELAX_MAX_STEPS, logfile=sys.stdout
        )
        if not converged:
            raise RuntimeError(f"Optimization did not converge within {RELAX_MAX_STEPS} steps.")

        # Sync relaxed geometry back to the input Pymatgen structure (in place).
        relaxed_structure_pmg = AseAtomsAdaptor.get_structure(relaxed_atoms)
        structure.lattice = relaxed_structure_pmg.lattice
        for i in range(len(structure)):
            structure.sites[i].frac_coords = relaxed_structure_pmg.sites[i].frac_coords

        print(f"  Relaxation complete. Final Energy: {final_energy:.4f} eV")
        return final_energy
    except Exception as e:
        print(f"  ERROR: Relaxation failed. Error: {e}")
        return np.nan

def calculate_voltage(host_struct, host_E, ionated_struct, ionated_E, ion_symbol):
    """
    Calculates voltage relative to SHE.
    Formula: V = - ( E_ionated - E_host - n * E_bulk ) / ( n * z ) + SHE_SHIFT
    """
    # 1. Get Reference Energy
    bulk_E = BULK_ENERGIES.get(ion_symbol)
    if bulk_E is None:
        print(f"ERROR: Bulk energy for {ion_symbol} not found in BULK_ENERGIES.")
        return np.nan

    # 2. Get Valency
    z = VALENCIES.get(ion_symbol, 1.0)

    # 3. Check inputs
    if pd.isna(host_E) or pd.isna(ionated_E):
        return np.nan

    # 4. Calculate number of ions (n)
    n_ions = ionated_struct.composition.get_el_amt_dict().get(ion_symbol, 0)
    if n_ions == 0:
        print(f"ERROR: No {ion_symbol} found in ionated structure.")
        return np.nan

    # 5. Calculate Voltage
    # Energy change of reaction: Delta G = E_ionated - E_host - n*E_bulk
    # Voltage = - Delta G / (n * z * e)  -- e is 1 in eV units
    reaction_energy = ionated_E - host_E - (n_ions * bulk_E)
    voltage = -1 * reaction_energy / (n_ions * z)

    # 6. Shift reference w.r.t. SHE
    voltage += SHE_SHIFT[ion_symbol]
    
    return voltage

def process_structure(filepath: str, calc: Optional['FAIRChemCalculator'] = None) -> Dict[str, Any]:
    
    # Initialize dynamic keys
    results = {
        f"{TARGET_ION}_relaxed_energy": np.nan,
        f"{REPLACEMENT_ION}_relaxed_energy": np.nan,
        "host_relaxed_energy": np.nan,
        f"{TARGET_ION}_voltage": np.nan,
        f"{REPLACEMENT_ION}_voltage": np.nan,
        "host_structure": None,
        f"{TARGET_ION}_structure": None,
        f"{REPLACEMENT_ION}_structure": None,
        "Success": False
    }

    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
    warnings.filterwarnings("ignore", "Atoms.*Defaulting to pbc=False", RuntimeWarning)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1. Load Structure ---
    print(f"Loading structure from: {filepath}")
    try:
        original_structure = Structure.from_file(filepath)
    except Exception as e:
        print(f"ERROR: Could not read structure file. Error: {e}")
        return results

    print(f"Original composition: {original_structure.composition.reduced_formula}")

    if TARGET_ION not in original_structure.composition:
        print(f"ERROR: Target ion ({TARGET_ION}) not found in structure. Exiting.")
        return results
        
    if calc is None:
        calc = get_ml_potential()

    base_name = os.path.splitext(os.path.basename(filepath))[0]

    # --- 2. Relax Target Structure (e.g. Mg) ---
    print(f"\nStep 1: Relaxing original {TARGET_ION} structure...")
    target_struct = original_structure.copy()
    target_E = get_relaxed_energy(target_struct, calc)
    results[f"{TARGET_ION}_structure"] = target_struct
    results[f"{TARGET_ION}_relaxed_energy"] = target_E
    
    # if not pd.isna(target_E):
    #     target_struct.to(filename=os.path.join(OUT_DIR, f"{base_name}_{TARGET_ION}.cif"))

    # --- 3. Swap to Replacement Ion (e.g. Li) and Relax ---
    print(f"\nStep 2: Replacing {TARGET_ION} with {REPLACEMENT_ION} and relaxing...")
    replacement_struct = original_structure.copy()
    replacement_struct.replace_species({TARGET_ION: REPLACEMENT_ION})
    
    replace_E = get_relaxed_energy(replacement_struct, calc)
    results[f"{REPLACEMENT_ION}_structure"] = replacement_struct
    results[f"{REPLACEMENT_ION}_relaxed_energy"] = replace_E

    # if not pd.isna(replace_E):
    #     replacement_struct.to(filename=os.path.join(OUT_DIR, f"{base_name}_{REPLACEMENT_ION}.cif"))

    # --- 4. Remove Target Ion to create Host ---
    print(f"\nStep 3: Removing {TARGET_ION} to create host...")
    host_struct = original_structure.copy()
    host_struct.remove_species([TARGET_ION])
    
    if len(host_struct) == 0:
        print("ERROR: Host structure is empty.")
        host_E = np.nan
    else:
        host_E = get_relaxed_energy(host_struct, calc)
        results["host_structure"] = host_struct
        results["host_relaxed_energy"] = host_E
        
        # if not pd.isna(host_E):
        #     host_struct.to(filename=os.path.join(OUT_DIR, f"{base_name}_host.cif"))

    # --- 5. Calculate Voltages ---
    if not any(pd.isna([target_E, replace_E, host_E])):
        print("\nCalculating voltages...")
        
        v_target = calculate_voltage(host_struct, host_E, target_struct, target_E, TARGET_ION)
        v_replace = calculate_voltage(host_struct, host_E, replacement_struct, replace_E, REPLACEMENT_ION)
        
        results[f"{TARGET_ION}_voltage"] = v_target
        results[f"{REPLACEMENT_ION}_voltage"] = v_replace
        results["Success"] = True

        print(f"  {TARGET_ION} Voltage: {v_target:.4f} V")
        print(f"  {REPLACEMENT_ION} Voltage: {v_replace:.4f} V")
    else:
        print("One or more relaxations failed, skipping voltage calc.")

    return results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <path_to_structure_file>")
        sys.exit(1)
    
    process_structure(sys.argv[1])
