import os
import time
import sys
import fire
import pandas as pd
import numpy as np
import contextlib
from typing import List, Dict, Any, Optional, Literal
from pymatgen.core import Structure
import json
from pathlib import Path
import torch

from mattergen.evaluation.evaluate import evaluate
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DefaultOrderedStructureMatcher,
)

# --- IMPORT THE POST_PROCESS MODULE DIRECTLY ---
from mattergen.scripts import post_process

# --- Dynamic Configuration from post_process ---
# We read these variables so the CSV headers match your post_process settings automatically
TARGET_ION = post_process.TARGET_ION          # e.g., "Mg"
REPLACEMENT_ION = post_process.REPLACEMENT_ION # e.g., "Li"

# --- Constants ---
OUT_DIR = post_process.OUT_DIR # Use the same output dir as post_process
OUTPUT_CSV_FILE = "voltage_analysis.csv"

# Define columns dynamically based on the ions
FINAL_COLUMN_NAMES = [
    'pretty_formula', 'filename', 'num_atoms',
    'host_energy', f'{TARGET_ION}_energy', f'{REPLACEMENT_ION}_energy',
    'host_energy_per_atom', f'{TARGET_ION}_energy_per_atom', f'{REPLACEMENT_ION}_energy_per_atom',
    f'{TARGET_ION}_voltage', f'{REPLACEMENT_ION}_voltage',
    'host_structure', f'{TARGET_ION}_structure', f'{REPLACEMENT_ION}_structure'
]

def get_cif_metadata(filepath: str) -> Dict[str, str]:
    """Extracts 'pretty_formula' and uses filename for 'filename'."""
    try:
        structure = Structure.from_file(filepath)
        pretty_formula = structure.composition.reduced_formula
        filename_base = os.path.splitext(os.path.basename(filepath))[0]
        return {"pretty_formula": pretty_formula, "filename": filename_base}
    except Exception as e:
        print(f"WARNING: Could not parse CIF file {filepath} for metadata. Error: {e}")
        return {
            "pretty_formula": os.path.basename(filepath),
            "filename": os.path.splitext(os.path.basename(filepath))[0],
        }

def structure_to_cif_string(struct: Optional[Structure]) -> str:
    """Converts a pymatgen Structure object to a multi-line CIF string."""
    if struct is None:
        return ""
    try:
        return struct.to(fmt="cif")
    except Exception:
        return "" 

def main(
    directory_path: str,
    relax: bool = False,
    structure_matcher: Literal["ordered", "disordered"] = "disordered",
    save_as: str | None = None,
    potential_load_path: (
        Literal["MatterSim-v1.0.0-1M.pth", "MatterSim-v1.0.0-5M.pth"] | None
    ) = None,
    reference_dataset_path: str | None = None,
    structures_output_path: str | None = None,
):
    """
    Main function to orchestrate the batch processing with Global Calculator.
    """
    
    # 1. Find all CIF files and apply numerical sorting
    if not os.path.exists(directory_path):
        print(f"ERROR: Directory {directory_path} does not exist.")
        return

    cif_names = [f for f in os.listdir(directory_path) if f.endswith(".cif")]

    def numerical_sort_key(filename):
        try:
            return int(os.path.splitext(filename)[0])
        except ValueError:
            return filename

    cif_names.sort(key=numerical_sort_key)
    cif_files = [os.path.join(directory_path, f) for f in cif_names]
    
    if not cif_files:
        print(f"No .cif files found in directory: {directory_path}")
        return

    print(f"Found {len(cif_files)} CIF files to process.")
    print(f"Target: {TARGET_ION} | Replacement: {REPLACEMENT_ION}")
    
    # --- 2. LOAD CALCULATOR ONCE ---
    print("\n" + "="*50)
    print("INITIALIZING GLOBAL ML CALCULATOR (This happens only once)")
    print("="*50)
    try:
        # We call the function from the imported module
        global_calc = post_process.get_ml_potential()
    except Exception as e:
        print(f"FATAL: Could not load ML Potential. {e}")
        return

    # Initialize CSV file
    pd.DataFrame(columns=FINAL_COLUMN_NAMES).to_csv(OUTPUT_CSV_FILE, index=False, mode='w')
    print(f"\nInitialized output file: {OUTPUT_CSV_FILE}")
    
    # Ensure output directory exists for log files
    full_OUT_DIR = os.path.join(directory_path, OUT_DIR)
    os.makedirs(full_OUT_DIR, exist_ok=True)

    # create dataframe of number of rows as length of cif_files and columns as FINAL_COLUMN_NAMES
    df = pd.DataFrame(columns=FINAL_COLUMN_NAMES, index=range(len(cif_files)))

    # 3. Process each file
    for i,filepath in enumerate(cif_files):
        start = time.time()

        base_name = os.path.splitext(os.path.basename(filepath))[0]
        log_filepath = os.path.join(full_OUT_DIR, f"{base_name}.log")
        
        print(f"-> Running analysis for: {base_name}...")
        
        # Capture metadata before processing
        metadata = get_cif_metadata(filepath)
        
        # --- EXECUTE PROCESSING WITH REDIRECTED LOGGING ---
        try:
            with open(log_filepath, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                # Pass the global_calc here!
                results = post_process.process_structure(filepath, calc=global_calc)
        except Exception as e:
            print(f"   !!! ERROR: Python execution failed for {base_name}. Error: {e}")
            results = {"Success": False}

        # Extract structures from results dictionary
        host_struct = results.get("host_structure")
        repl_struct = results.get(f"{REPLACEMENT_ION}_structure")
        targ_struct = results.get(f"{TARGET_ION}_structure")

        # Calculate lengths/metrics
        host_len = len(host_struct) if host_struct else np.nan
        repl_len = len(repl_struct) if repl_struct else np.nan
        targ_len = len(targ_struct) if targ_struct else np.nan

        # Calculate Normalized Energies (per atom)
        # Note: Host has fewer atoms, so we need to be careful with normalization if used for comparison
        # Here we normalize by the number of atoms in that specific phase.
        # If host structure isn't loaded, we can't get exact host len easily without reading that file too.
        # For now, we approximate per_atom based on the replacement structure length 
        # (assuming standard substitution).
        
        final_row_data = {
            "pretty_formula": metadata["pretty_formula"],
            "filename": metadata["filename"],
            "num_atoms": repl_len,
            
            "host_energy": results.get("host_relaxed_energy") if host_struct else np.nan,
            f"{REPLACEMENT_ION}_energy": results.get(f"{REPLACEMENT_ION}_relaxed_energy") if repl_struct else np.nan,
            f"{TARGET_ION}_energy": results.get(f"{TARGET_ION}_relaxed_energy") if targ_struct else np.nan,
            
            "host_energy_per_atom": np.divide(results.get("host_relaxed_energy"), host_len) if host_struct else np.nan,
            f"{REPLACEMENT_ION}_energy_per_atom": np.divide(results.get(f"{REPLACEMENT_ION}_relaxed_energy"), repl_len) if repl_struct else np.nan,
            f"{TARGET_ION}_energy_per_atom": np.divide(results.get(f"{TARGET_ION}_relaxed_energy"), targ_len) if targ_struct else np.nan,

            f"{REPLACEMENT_ION}_voltage": results.get(f"{REPLACEMENT_ION}_voltage"),
            f"{TARGET_ION}_voltage": results.get(f"{TARGET_ION}_voltage"),
            "host_structure": host_struct,
            f"{REPLACEMENT_ION}_structure": repl_struct,
            f"{TARGET_ION}_structure": targ_struct,
        }

        end = time.time()
        
        # Create DataFrame row and Append
        df_row = pd.DataFrame([final_row_data])

        df.iloc[i] = df_row.iloc[0]
        df_row["host_structure"] = df_row["host_structure"].apply(structure_to_cif_string)
        df_row[f"{REPLACEMENT_ION}_structure"] = df_row[f"{REPLACEMENT_ION}_structure"].apply(structure_to_cif_string)
        df_row[f"{TARGET_ION}_structure"] = df_row[f"{TARGET_ION}_structure"].apply(structure_to_cif_string)
        df_row = df_row.reindex(columns=FINAL_COLUMN_NAMES)
        df_row.to_csv(OUTPUT_CSV_FILE, index=False, mode='a', header=False)
        
        print(f"    Result appended. (Log saved to {log_filepath})", flush=True)
        
    print(f"\n DONE! All results saved incrementally to **{OUTPUT_CSV_FILE}**")
    
    # --- 4. POST-PROCESSING EVALUATION METRICS ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = df.dropna(subset=["Li_structure", "Li_energy"])
    structures = df["Li_structure"].tolist()
    energies = df["Li_energy"].tolist()
    structure_matcher = (
        DefaultDisorderedStructureMatcher()
        if structure_matcher == "disordered"
        else DefaultOrderedStructureMatcher()
    )
    reference = None
    if reference_dataset_path:
        reference = LMDBGZSerializer().deserialize(reference_dataset_path)

    # When relax=True, evaluate() re-relaxes with UMA and computes energies itself,
    # so the pre-computed UMA energies must not be passed (they are mutually exclusive).
    metrics = evaluate(
        structures=structures,
        relax=relax,
        energies=(None if relax else energies),
        structure_matcher=structure_matcher,
        save_as=save_as,
        potential_load_path=potential_load_path,
        reference=reference,
        device=device,
        structures_output_path=structures_output_path,
    )
    print(json.dumps(metrics, indent=2))


def _main():
    fire.Fire(main)


if __name__ == "__main__":
    _main()
