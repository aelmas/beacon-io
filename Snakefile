"""Snakemake workflow for BEACON-IO: Expression-driven dependencies at the
tumor-immune interface for precision immuno-oncology.

DAG
───
  01_download → 02_beacon_edd
                     │
         ┌───────────┼────────────────┐
         ▼           ▼                ▼
   03_immune    04_combination   06_singlecell
         │           │                │
         └───────────┼────────────────┘
                     ▼
            05_clinical_validation
                     │
                     ▼
              07_integration

Steps 03, 04, and 06 are independent and run in parallel.
Step 05 depends on 03 (needs immune stratification for signature building).
Step 07 integrates all upstream outputs.

Usage
─────
    snakemake --cores 8                          # Full pipeline
    snakemake integration                        # Final integration (+ all deps)
    snakemake beacon_edd                         # BEACON only (+ download)
    snakemake -n                                 # Dry run
    snakemake --config beacon_mode=fast           # Fast Spearman screen only
    snakemake --config skip_singlecell=True       # Skip scRNA-seq analysis
"""

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT = config.get("output_dir", "analysis/out")
DATA   = config.get("data_dir",   "data/raw")

BEACON_MODE    = config.get("beacon_mode", "both")    # fast | full | both
SKIP_SC        = config.get("skip_singlecell", False)
SKIP_CLINICAL  = config.get("skip_clinical", False)


# ── Final target ──────────────────────────────────────────────────────────────

rule all:
    """Default target: produce the integrated BEACON-IO evidence table."""
    input:
        f"{OUTPUT}/integration/beacon_io_evidence_table.csv",


# ── Step 01: Download all datasets ───────────────────────────────────────────

rule download:
    """Download DepMap, PRISM, TCGA, and ICB cohort data."""
    output:
        stamp = touch(f"{OUTPUT}/.download_done"),
    log:
        f"{OUTPUT}/logs/01_download.log",
    shell:
        "python scripts/01_download_data.py --data-dir {DATA} 2>&1 | tee {log}"


# ── Step 02: BEACON expression-driven dependency ─────────────────────────────

rule beacon_edd:
    """Run BEACON analysis across all lineages."""
    input:
        f"{OUTPUT}/.download_done",
    output:
        significant = f"{OUTPUT}/beacon_edd/beacon_edd_significant.csv",
        all_results = f"{OUTPUT}/beacon_edd/beacon_edd_all_lineages.csv",
        stamp       = touch(f"{OUTPUT}/.beacon_done"),
    log:
        f"{OUTPUT}/logs/02_beacon_edd.log",
    threads: 4
    shell:
        """
        python scripts/02_beacon_edd.py \
            --data-dir {DATA} \
            --output-dir {OUTPUT} \
            --mode {BEACON_MODE} \
            2>&1 | tee {log}
        """


# ── Step 03: Immune contextualization ────────────────────────────────────────

rule immune_context:
    """TME deconvolution, immune evasion scoring, differential EDD."""
    input:
        f"{OUTPUT}/.beacon_done",
    output:
        diff_edd   = f"{OUTPUT}/immune_context/differential_edd_hot_vs_cold.csv",
        evasion    = f"{OUTPUT}/immune_context/evasion_correlated_edd.csv",
        stamp      = touch(f"{OUTPUT}/.immune_done"),
    log:
        f"{OUTPUT}/logs/03_immune_context.log",
    shell:
        """
        python scripts/03_immune_context.py \
            --data-dir {DATA} \
            --output-dir {OUTPUT} \
            2>&1 | tee {log}
        """


# ── Step 04: Combination target discovery ────────────────────────────────────

rule combination:
    """PRISM drug sensitivity + DGIdb druggability + combination scoring."""
    input:
        f"{OUTPUT}/.beacon_done",
    output:
        prism   = f"{OUTPUT}/combination/prism_edd_drug_hits.csv",
        combos  = f"{OUTPUT}/combination/icb_combination_candidates.csv",
        stamp   = touch(f"{OUTPUT}/.combination_done"),
    log:
        f"{OUTPUT}/logs/04_combination.log",
    shell:
        """
        python scripts/04_combination_targets.py \
            --data-dir {DATA} \
            --output-dir {OUTPUT} \
            2>&1 | tee {log}
        """


# ── Step 05: Clinical validation ─────────────────────────────────────────────

rule clinical:
    """TCGA survival + ICB response prediction + biomarker benchmarking."""
    input:
        f"{OUTPUT}/.beacon_done",
        f"{OUTPUT}/.immune_done",
    output:
        survival  = f"{OUTPUT}/clinical/tcga_survival_results.csv",
        benchmark = f"{OUTPUT}/clinical/icb_biomarker_benchmark.csv",
        stamp     = touch(f"{OUTPUT}/.clinical_done"),
    log:
        f"{OUTPUT}/logs/05_clinical.log",
    shell:
        """
        python scripts/05_clinical_validation.py \
            --data-dir {DATA} \
            --output-dir {OUTPUT} \
            2>&1 | tee {log}
        """


# ── Step 06: Single-cell resolution ──────────────────────────────────────────

rule singlecell:
    """Cell-type resolution, compartment classification, sc-level DE."""
    input:
        f"{OUTPUT}/.beacon_done",
    output:
        compartment = f"{OUTPUT}/singlecell/compartment_consensus.csv",
        stamp       = touch(f"{OUTPUT}/.singlecell_done"),
    log:
        f"{OUTPUT}/logs/06_singlecell.log",
    shell:
        """
        python scripts/06_singlecell.py \
            --data-dir {DATA} \
            --output-dir {OUTPUT} \
            2>&1 | tee {log}
        """


# ── Step 07: Multi-evidence integration ──────────────────────────────────────

def integration_inputs(wildcards):
    """Gather all upstream outputs for integration."""
    inputs = [
        f"{OUTPUT}/.beacon_done",
        f"{OUTPUT}/.immune_done",
        f"{OUTPUT}/.combination_done",
    ]
    if not SKIP_CLINICAL:
        inputs.append(f"{OUTPUT}/.clinical_done")
    if not SKIP_SC:
        inputs.append(f"{OUTPUT}/.singlecell_done")
    return inputs


rule integration:
    """Compile all evidence into ranked BEACON-IO target catalogue."""
    input:
        integration_inputs,
    output:
        evidence = f"{OUTPUT}/integration/beacon_io_evidence_table.csv",
        top30    = f"{OUTPUT}/integration/beacon_io_top30_targets.csv",
    log:
        f"{OUTPUT}/logs/07_integration.log",
    shell:
        """
        python scripts/07_integration.py \
            --output-dir {OUTPUT} \
            2>&1 | tee {log}
        """
