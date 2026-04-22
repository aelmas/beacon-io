#!/usr/bin/env python3
"""Step 07: Multi-evidence integration — compile all results into ranked target list.

Reads outputs from steps 02-06 and produces:
  - Unified evidence table (gene x evidence_streams)
  - Composite-scored target ranking
  - Final BEACON-IO target catalogue
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beacon_io.config import CFG
from beacon_io.utils import ensure_dir, get_logger
from integration.evidence import compile_evidence, summarise_top_targets

log = get_logger("07_integration")


def _load_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    log.warning("Missing: %s", path)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=CFG["output_dir"])
    args = parser.parse_args()

    base = Path(args.output_dir)
    out = ensure_dir(base / "integration")

    # Load all evidence streams
    edd_results = _load_if_exists(base / "beacon_edd" / "beacon_edd_significant.csv")
    diff_edd = _load_if_exists(base / "immune_context" / "differential_edd_hot_vs_cold.csv")
    evasion_corr = _load_if_exists(base / "immune_context" / "evasion_correlated_edd.csv")
    prism_hits = _load_if_exists(base / "combination" / "prism_edd_drug_hits.csv")
    icb_meta = _load_if_exists(base / "clinical" / "icb_meta_analysis.csv")
    tcga_survival = _load_if_exists(base / "clinical" / "tcga_survival_results.csv")
    compartment = _load_if_exists(base / "singlecell" / "compartment_consensus.csv")
    druggability = _load_if_exists(base / "combination" / "dgidb_annotations.csv")

    # Compile
    log.info("=== Compiling multi-evidence target ranking ===")
    evidence = compile_evidence(
        edd_results=edd_results,
        diff_edd=diff_edd,
        evasion_corr=evasion_corr,
        prism_hits=prism_hits,
        icb_meta=icb_meta,
        tcga_survival=tcga_survival,
        compartment=compartment,
        druggability=druggability,
    )
    evidence.to_csv(out / "beacon_io_evidence_table.csv", index=False)
    log.info("Evidence table: %d genes scored", len(evidence))

    # Top targets
    top = summarise_top_targets(evidence, top_n=30)
    top.to_csv(out / "beacon_io_top30_targets.csv", index=False)
    log.info("Top 30 BEACON-IO targets:\n%s", top.to_string(index=False))

    # Summary statistics
    summary = {
        "total_edd_genes": len(edd_results) if not edd_results.empty else 0,
        "immune_specific_genes": (diff_edd["fdr"] < 0.05).sum() if not diff_edd.empty and "fdr" in diff_edd.columns else 0,
        "prism_drug_pairs": len(prism_hits) if not prism_hits.empty else 0,
        "druggable_targets": evidence["E8_druggable"].sum() if "E8_druggable" in evidence.columns else 0,
        "tumour_intrinsic": (evidence.get("E7_tumour_intrinsic", 0) == 1).sum(),
        "top_composite_score": evidence["composite_score"].max() if not evidence.empty else 0,
    }
    pd.Series(summary).to_csv(out / "summary_statistics.csv")
    log.info("Summary: %s", summary)

    log.info("Integration complete. Final catalogue: %s", out / "beacon_io_evidence_table.csv")


if __name__ == "__main__":
    main()
