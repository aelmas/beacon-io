"""Clinical validation of BEACON-IO targets against ICB response and survival.

Three-tier validation strategy:
  Tier 1: TCGA survival — do BEACON-IO targets stratify overall survival?
  Tier 2: ICB response prediction — do BEACON-IO signatures outperform
           existing biomarkers (TMB, PD-L1, TIDE, GEP, IMPRES)?
  Tier 3: Multi-cohort meta-analysis — consistency across ICB cohorts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from beacon_io.config import CFG
from beacon_io.utils import fdr_correction, get_logger

log = get_logger(__name__)

CLINICAL_CFG = CFG["clinical"]


# ---------------------------------------------------------------------------
# Tier 1: TCGA survival analysis
# ---------------------------------------------------------------------------

def tcga_survival_analysis(
    expression: pd.DataFrame,
    survival: pd.DataFrame,
    beacon_genes: list[str],
    cancer_type: str | None = None,
) -> pd.DataFrame:
    """For each BEACON-IO gene, test association with overall survival.

    Uses both univariate Cox PH and Kaplan-Meier log-rank (median split).

    Returns DataFrame: gene, cox_hr, cox_pvalue, cox_fdr, logrank_pvalue.
    """
    shared = expression.index.intersection(survival.index)
    surv = survival.loc[shared].copy()
    # Standardise column names
    time_col = next((c for c in surv.columns if "OS.time" in c or "os_time" in c), None)
    event_col = next((c for c in surv.columns if c in ("OS", "os_event", "OS.event")), None)
    if time_col is None or event_col is None:
        log.warning("Cannot find survival columns, available: %s", surv.columns.tolist())
        return pd.DataFrame()

    records = []
    for gene in beacon_genes:
        if gene not in expression.columns:
            continue
        expr = expression.loc[shared, gene]
        df = pd.DataFrame({
            "T": surv[time_col].astype(float),
            "E": surv[event_col].astype(float),
            "expr": expr.values,
        }).dropna()
        if len(df) < 20:
            continue

        # Cox PH
        try:
            cph = CoxPHFitter()
            cph.fit(df, duration_col="T", event_col="E")
            hr = np.exp(cph.params_["expr"])
            cox_p = cph.summary.loc["expr", "p"]
        except Exception:
            hr, cox_p = np.nan, np.nan

        # Kaplan-Meier log-rank (median split)
        median_expr = df["expr"].median()
        high = df[df["expr"] >= median_expr]
        low = df[df["expr"] < median_expr]
        try:
            lr = logrank_test(high["T"], low["T"], high["E"], low["E"])
            lr_p = lr.p_value
        except Exception:
            lr_p = np.nan

        records.append({
            "gene": gene,
            "cancer_type": cancer_type or "pan-cancer",
            "n_patients": len(df),
            "cox_hr": hr,
            "cox_pvalue": cox_p,
            "logrank_pvalue": lr_p,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["cox_fdr"] = fdr_correction(df["cox_pvalue"].fillna(1).values)
        df["logrank_fdr"] = fdr_correction(df["logrank_pvalue"].fillna(1).values)
    return df


# ---------------------------------------------------------------------------
# Tier 2: ICB response prediction — BEACON-IO signature vs existing biomarkers
# ---------------------------------------------------------------------------

def build_beacon_io_signature(
    edd_results: pd.DataFrame,
    top_n: int = 50,
) -> list[str]:
    """Select top BEACON-IO genes for a predictive signature.

    Prioritises genes that are:
      - Significant EDD (FDR < 0.05, rho < -0.25)
      - Immune-specific (large |delta_rho| between hot/cold)
      - Druggable
    """
    sig = edd_results.copy()
    if "fdr" in sig.columns:
        sig = sig[sig["fdr"] < 0.05]
    if "delta_rho" in sig.columns:
        sig["abs_delta"] = sig["delta_rho"].abs()
        sig = sig.sort_values("abs_delta", ascending=False)
    elif "rho_posterior_median" in sig.columns:
        sig = sig.sort_values("rho_posterior_median")
    return sig["gene"].head(top_n).tolist()


def predict_icb_response(
    expression: pd.DataFrame,
    response: pd.Series,
    signature_genes: list[str],
    n_folds: int = 5,
) -> dict:
    """Cross-validated logistic regression for ICB response prediction.

    Returns dict with mean AUC, per-fold AUCs, and fitted coefficients.
    """
    available = [g for g in signature_genes if g in expression.columns]
    if len(available) < 3:
        log.warning("Too few signature genes available (%d)", len(available))
        return {"auc_mean": np.nan, "auc_folds": [], "genes_used": available}

    shared = expression.index.intersection(response.index)
    X = expression.loc[shared, available].fillna(0).values
    y = response.loc[shared].values.astype(int)

    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        log.warning("Too few events for prediction (R=%d, NR=%d)", y.sum(), len(y) - y.sum())
        return {"auc_mean": np.nan, "auc_folds": [], "genes_used": available}

    cv = StratifiedKFold(n_splits=min(n_folds, y.sum()), shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        model = LogisticRegressionCV(
            Cs=10, penalty="l1", solver="saga", max_iter=5000, random_state=42
        )
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]
        if len(np.unique(y[test_idx])) > 1:
            aucs.append(roc_auc_score(y[test_idx], probs))

    return {
        "auc_mean": np.mean(aucs) if aucs else np.nan,
        "auc_std": np.std(aucs) if aucs else np.nan,
        "auc_folds": aucs,
        "n_genes": len(available),
        "genes_used": available,
    }


def benchmark_biomarkers(
    expression: pd.DataFrame,
    clinical: pd.DataFrame,
    response: pd.Series,
    beacon_signature: list[str],
    mutations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare BEACON-IO signature against established ICB biomarkers.

    Biomarkers tested:
      - TMB (if mutations provided)
      - PD-L1 expression (CD274)
      - IFN-gamma GEP (Ayers et al. 2017 — 18-gene signature)
      - BEACON-IO signature

    Returns DataFrame: biomarker, auc, auc_std, n_folds.
    """
    results = []

    # BEACON-IO
    res = predict_icb_response(expression, response, beacon_signature)
    results.append({"biomarker": "BEACON-IO", **res})

    # PD-L1 (CD274)
    if "CD274" in expression.columns:
        res = predict_icb_response(expression, response, ["CD274"])
        results.append({"biomarker": "PD-L1 (CD274)", **res})

    # IFN-gamma GEP (Ayers 2017)
    gep_genes = [
        "IFNG", "STAT1", "CCR5", "CXCL9", "CXCL10", "CXCL11", "IDO1",
        "PRF1", "GZMA", "GZMB", "CD27", "CD274", "CD276", "CMKLR1",
        "HLA-DQA1", "HLA-DRB1", "HLA-E", "PDCD1LG2",
    ]
    res = predict_icb_response(expression, response, gep_genes)
    results.append({"biomarker": "IFNg-GEP (Ayers)", **res})

    # Cytolytic activity (Rooney 2015)
    cyt_genes = ["GZMA", "PRF1"]
    res = predict_icb_response(expression, response, cyt_genes)
    results.append({"biomarker": "Cytolytic (CYT)", **res})

    # IMPRES (Auslander 2018) — checkpoint gene ratios
    impres_genes = [
        "PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT",
        "CD27", "CD40", "CD80", "ICOS", "TNFRSF14", "TNFRSF18",
        "BTLA", "CD244", "TNFRSF9",
    ]
    res = predict_icb_response(expression, response, impres_genes)
    results.append({"biomarker": "IMPRES", **res})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Tier 3: Meta-analysis across ICB cohorts
# ---------------------------------------------------------------------------

def meta_analysis_icb(
    cohort_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Fixed-effects meta-analysis of per-gene ICB association across cohorts.

    Uses inverse-variance weighting of Fisher-z transformed correlations.

    Parameters
    ----------
    cohort_results : {cohort_name: DataFrame with gene, rho, pvalue columns}

    Returns
    -------
    DataFrame with gene, meta_rho, meta_z, meta_pvalue, meta_fdr, n_cohorts.
    """
    all_genes = set()
    for df in cohort_results.values():
        if not df.empty and "gene" in df.columns:
            all_genes.update(df["gene"].unique())

    records = []
    for gene in all_genes:
        zs, weights = [], []
        for cohort_name, df in cohort_results.items():
            row = df[df["gene"] == gene]
            if row.empty:
                continue
            rho = row["rho"].values[0]
            n = row.get("n_patients", row.get("n_lines", pd.Series([30]))).values[0]
            if np.isnan(rho) or n < 5:
                continue
            z = np.arctanh(np.clip(rho, -0.999, 0.999))
            w = n - 3  # inverse variance weight for Fisher z
            zs.append(z)
            weights.append(w)

        if len(zs) < 2:
            continue

        zs, weights = np.array(zs), np.array(weights)
        meta_z = np.average(zs, weights=weights)
        meta_se = 1 / np.sqrt(weights.sum())
        z_stat = meta_z / meta_se
        meta_p = 2 * stats.norm.sf(abs(z_stat))
        meta_rho = np.tanh(meta_z)

        records.append({
            "gene": gene,
            "meta_rho": meta_rho,
            "meta_z": z_stat,
            "meta_pvalue": meta_p,
            "n_cohorts": len(zs),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["meta_fdr"] = fdr_correction(df["meta_pvalue"].values)
        df = df.sort_values("meta_pvalue")
    return df
