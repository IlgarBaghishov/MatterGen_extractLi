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

from mattergen.evaluation.evaluate import evaluate, evaluate_per_structure
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DefaultOrderedStructureMatcher,
)


OUTPUT_CSV_FILE = "merged_data.csv"

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
    
def cif_string_to_structure(cif_string: str) -> Optional[Structure]:
    """Converts a CIF string back to a pymatgen Structure object."""
    if not cif_string.strip():
        return None
    try:
        return Structure.from_str(cif_string, fmt="cif")
    except Exception:
        return None

def main(
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
    Main function to orchestrate the SUN evaluation similarly to what MatterGen does.
    """
    save_as = OUTPUT_CSV_FILE[:-4]+"_evaluation_metrics.json" if save_as is None else save_as
    df = pd.read_csv(OUTPUT_CSV_FILE)
    df = df.dropna(subset=["cif", "Li_energy_per_atom"])
    df['Li_structure_pymatgen'] = df['cif'].apply(cif_string_to_structure)

    # --- 4. POST-PROCESSING EVALUATION METRICS ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    structures = df["Li_structure_pymatgen"].tolist()
    # get total energy from Li_energy_per_atom and number of atoms in the structure which is found in Li_structure_pymatgen column
    df["Li_energy"] = df.apply(lambda row: row["Li_energy_per_atom"] * len(row["Li_structure_pymatgen"]), axis=1)
    energies = df["Li_energy"].tolist()
    structure_matcher = (
        DefaultDisorderedStructureMatcher()
        if structure_matcher == "disordered"
        else DefaultOrderedStructureMatcher()
    )
    reference = None
    if reference_dataset_path:
        reference = LMDBGZSerializer().deserialize(reference_dataset_path)

    evaluator = evaluate_per_structure(
        structures=structures,
        relax=relax,
        energies=energies,
        structure_matcher=structure_matcher,
        potential_load_path=potential_load_path,
        reference=reference,
        device=device,
        structures_output_path=structures_output_path,
    )

    # Build per-structure DataFrame
    # result_df = df[['pretty_formula', 'filename']].copy()
    df['energy_above_hull'] = evaluator.energy_capability.energy_above_hull
    df['unique_group_id'] = evaluator.unique_group_ids
    df['is_novel'] = evaluator.is_novel.astype(int)
    df.to_csv(OUTPUT_CSV_FILE[:-4]+"_SUN.csv", index=False)

    # Still compute and print summary metrics
    metrics = evaluator.compute_metrics(
        metrics=evaluator.available_metrics,
        save_as=save_as,
        pretty_print=True,
    )
    print(json.dumps(metrics, indent=2))


def _main():
    fire.Fire(main)


if __name__ == "__main__":
    _main()
