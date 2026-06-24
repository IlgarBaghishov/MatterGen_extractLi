"""Create a `_diff` version of a voltage dataset (the "ΔV trick").

A `_diff` dataset is identical to its source except the **`Na_voltage` column is
overwritten with the Li/Na voltage difference**:

    Na_voltage  <-  Li_voltage - Na_voltage          (Li_voltage is left unchanged)

so that a model "conditioned on Na_voltage" is really conditioned on
ΔV = V_Li - V_Na. See CLAUDE.md ("The ΔV trick") for the rationale.

Usage:
    python -m mattergen.scripts.make_diff_dataset --src_folder datasets/li_data_v2
    # writes datasets/li_data_v2_diff/{train,val,test}.csv
or, after `pip install -e .`:
    mattergen-make-diff --src_folder datasets/li_data_v2 --dst_folder datasets/li_data_v2_diff
"""

import os

import fire
import pandas as pd


def make_diff(
    src_folder: str,
    dst_folder: str | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    """Write a `_diff` copy of the CSV splits in ``src_folder``.

    Args:
        src_folder: folder containing train.csv / val.csv / test.csv with absolute
            ``Li_voltage`` and ``Na_voltage`` columns.
        dst_folder: output folder (default: ``<src_folder>_diff``).
        splits: which split files to convert.
    """
    src_folder = src_folder.rstrip("/")
    if dst_folder is None:
        dst_folder = src_folder + "_diff"
    os.makedirs(dst_folder, exist_ok=True)

    for split in splits:
        src = os.path.join(src_folder, f"{split}.csv")
        if not os.path.exists(src):
            print(f"skip (missing): {src}")
            continue
        df = pd.read_csv(src)
        for col in ("Li_voltage", "Na_voltage"):
            if col not in df.columns:
                raise ValueError(f"{src} is missing required column '{col}'")
        df = df.copy()
        # ΔV trick: overwrite Na_voltage with the difference; keep Li_voltage as-is.
        df["Na_voltage"] = df["Li_voltage"] - df["Na_voltage"]
        dst = os.path.join(dst_folder, f"{split}.csv")
        df.to_csv(dst, index=False)
        print(f"wrote {dst}  ({len(df)} rows) | Na_voltage = Li_voltage - Na_voltage")

    print(f"DONE -> {dst_folder}")


def _main() -> None:
    fire.Fire(make_diff)


if __name__ == "__main__":
    _main()
