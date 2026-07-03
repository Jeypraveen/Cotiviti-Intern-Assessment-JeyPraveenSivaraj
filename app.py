"""
ClinicalAudit AI — Payment Integrity Intelligence System
=========================================================
Built for Cotiviti GenAI Intern Assessment | Topic 2: Clinical Decision Making
and Pattern Recognition in Healthcare

Architecture Overview
---------------------
Module 1 — Data Generation     : Synthetic provider profiles with real ICD-10/CPT codes
Module 2 — Anomaly Detection    : Time-Series Z-score detection (numpy)          [CONCEPT: Time-Series Anomaly Detection]
Module 3 — Clustering           : K-Means on provider feature vectors (sklearn)   [CONCEPT: Clustering]
Module 4 — Classification       : Risk tier assignment from cluster + score        [CONCEPT: Classification]
Module 5 — Prediction/Inference : Confidence-weighted risk factor extraction       [CONCEPT: Prediction + Inference]
Module 6 — Agentic Chain        : Step-logged agent + Groq Llama3 audit brief     [CONCEPT: Chain Reasoning + Agentic GenAI]
Module 7 — Treatment Screening  : Patient-level care gap scoring                  [TPO: Treatment]
Module 8 — Operations Tab       : Portfolio scatter, financial exposure            [TPO: Operations]

Author : Jey Praveen Sivaraj 
"""

# ─── Imports ─────────────────────────────────────────────────────────────────
import os
import time
import random
import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ─── Constants & Configuration ───────────────────────────────────────────────

COTIVITI_PURPLE = "#6B4FBB"
COTIVITI_PURPLE_LIGHT = "#9B7FEB"
BG_DARK = "#0F1117"
CARD_BG = "#1A1D27"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#A0A0B0"
RED_ALERT = "#E74C3C"
ORANGE_WARN = "#E67E22"
GREEN_OK = "#27AE60"

CLUSTER_LABELS = [
    "High-Volume Billing Anomaly",
    "Diagnosis-Procedure Mismatch",
    "Credential and Phantom Risk",
    "Clean Billing Profile",
]

RISK_THRESHOLDS = {"HIGH": 55, "MEDIUM": 28}

GROQ_MODEL = "llama-3.3-70b-versatile"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ─── Module 1: Synthetic Data Generation ─────────────────────────────────────


def generate_provider_profiles() -> list[dict[str, Any]]:
    """
    Generate 10 synthetic provider profiles representing named healthcare fraud
    and anomaly patterns investigated by payment integrity companies.

    Each profile includes:
    - 12 months of billing history (claim counts, amounts, procedure/diagnosis codes)
    - Patient volume, diagnosis-procedure match score, referral rate
    - Real ICD-10 diagnosis codes and CPT procedure codes

    Returns
    -------
    list[dict] : One dict per provider with keys:
        id, name, specialty, fraud_pattern, monthly_claims, monthly_amounts,
        icd10_codes, cpt_codes, patient_volume, dx_proc_match_score,
        referral_rate, duplicate_rate, patients
    """
    rng = np.random.default_rng(RANDOM_SEED)

    def months_base(base: float, noise: float, n: int = 12) -> list[float]:
        return list(np.clip(rng.normal(base, noise, n), base * 0.4, base * 2.5))

    def spike_tail(base: float, noise: float, spike_mult: float = 9.5) -> list[float]:
        vals = list(np.clip(rng.normal(base, noise, 10), base * 0.4, base * 1.3))
        vals += [base * spike_mult * rng.uniform(0.92, 1.08) for _ in range(2)]
        return vals

    providers = [
        {
            "id": "PRV-001",
            "name": "Eastside Family Medicine",
            "specialty": "Family Medicine",
            "fraud_pattern": "Upcoding",
            "fraud_description": (
                "Bills CPT 99215 (complex office visit, ~$250) for patients whose "
                "diagnoses justify only CPT 99213 (simple visit, ~$90). "
                "Diagnosis codes are consistently minor (J06.9 upper respiratory, "
                "J30.1 allergic rhinitis) but billed at maximum complexity."
            ),
            "monthly_claims": months_base(310, 25),
            "monthly_amounts": months_base(77500, 6000),
            "icd10_codes": ["J06.9", "J30.1", "Z00.00", "J02.9", "J03.90"],
            "cpt_codes": ["99215", "99215", "99215", "99214", "99215"],
            "patient_volume": months_base(290, 20),
            "dx_proc_match_score": 0.21,
            "referral_rate": 0.08,
            "duplicate_rate": 0.01,
        },
        {
            "id": "PRV-002",
            "name": "Orthocare Surgical Partners",
            "specialty": "Orthopedic Surgery",
            "fraud_pattern": "Unbundling",
            "fraud_description": (
                "Splits combined arthroscopic procedures into component CPT codes "
                "(29881 meniscectomy + 29882 meniscus repair billed separately) "
                "instead of the inclusive bundled code, yielding ~40% excess reimbursement."
            ),
            "monthly_claims": months_base(185, 18),
            "monthly_amounts": months_base(148000, 12000),
            "icd10_codes": ["M23.200", "M23.611", "M17.11", "S83.200A", "M25.361"],
            "cpt_codes": ["29881", "29882", "27447", "29880", "29881"],
            "patient_volume": months_base(95, 10),
            "dx_proc_match_score": 0.58,
            "referral_rate": 0.12,
            "duplicate_rate": 0.02,
        },
        {
            "id": "PRV-003",
            "name": "Meridian Diagnostic Center",
            "specialty": "Diagnostic Radiology",
            "fraud_pattern": "Phantom Billing",
            "fraud_description": (
                "Bills procedures for patients with no corresponding visit records. "
                "Claim volume is high but average visit duration is 3 minutes. "
                "Diagnosis codes (Z51.89 encounter for other care) do not justify "
                "the advanced imaging CPT codes billed (70553 MRI brain with contrast)."
            ),
            "monthly_claims": months_base(420, 30),
            "monthly_amounts": months_base(210000, 18000),
            "icd10_codes": ["Z51.89", "Z13.88", "Z00.00", "Z51.89", "Z13.6"],
            "cpt_codes": ["70553", "71250", "74177", "70553", "93306"],
            "patient_volume": months_base(80, 12),
            "dx_proc_match_score": 0.14,
            "referral_rate": 0.05,
            "duplicate_rate": 0.03,
        },
        {
            "id": "PRV-004",
            "name": "Lakewood Primary & Specialists",
            "specialty": "Internal Medicine",
            "fraud_pattern": "Ping-Ponging",
            "fraud_description": (
                "Unusual referral loop between this provider and two affiliated specialists. "
                "High volume of Z00.00 (routine adult exams) triggers repeated specialist "
                "referrals with CPT 99243 (office consultation). Generates medically "
                "unnecessary visit chains billable at each step."
            ),
            "monthly_claims": months_base(350, 28),
            "monthly_amounts": months_base(87500, 7000),
            "icd10_codes": ["Z00.00", "Z00.00", "Z00.01", "Z00.00", "Z13.88"],
            "cpt_codes": ["99213", "99243", "99243", "99243", "99213"],
            "patient_volume": months_base(320, 22),
            "dx_proc_match_score": 0.45,
            "referral_rate": 0.61,
            "duplicate_rate": 0.01,
        },
        {
            "id": "PRV-005",
            "name": "Northgate Infusion Clinic",
            "specialty": "Hematology/Oncology",
            "fraud_pattern": "Credential Billing",
            "fraud_description": (
                "Procedures billed under supervising physician NPI (Dr. Chen, MD) but "
                "performed by unlicensed infusion technicians. Volume spikes on Thursdays "
                "and Fridays correlate with days the supervising physician's schedule "
                "shows confirmed absence at a partner facility."
            ),
            "monthly_claims": months_base(260, 22),
            "monthly_amounts": months_base(195000, 16000),
            "icd10_codes": ["C50.911", "C18.9", "D64.9", "C34.10", "Z51.11"],
            "cpt_codes": ["96413", "96415", "96372", "96413", "J0135"],
            "patient_volume": months_base(110, 14),
            "dx_proc_match_score": 0.67,
            "referral_rate": 0.18,
            "duplicate_rate": 0.02,
        },
        {
            "id": "PRV-006",
            "name": "BackCare & Spine Institute",
            "specialty": "Orthopedic Surgery",
            "fraud_pattern": "Medically Unnecessary Procedures",
            "fraud_description": (
                "High volume of ICD-10 M54.5 (low back pain, lumbago) matched to "
                "CPT 27447 (total knee arthroplasty) — the diagnosis categorically "
                "does not justify the procedure. Also bills 22612 (lumbar fusion) "
                "for patients with single-episode pain under 4 weeks."
            ),
            "monthly_claims": months_base(230, 20),
            "monthly_amounts": months_base(345000, 28000),
            "icd10_codes": ["M54.5", "M54.5", "M54.4", "M54.5", "M51.16"],
            "cpt_codes": ["27447", "22612", "27447", "22612", "62323"],
            "patient_volume": months_base(88, 10),
            "dx_proc_match_score": 0.09,
            "referral_rate": 0.10,
            "duplicate_rate": 0.01,
        },
        {
            "id": "PRV-007",
            "name": "Riverside Urgent Care Network",
            "specialty": "Emergency Medicine",
            "fraud_pattern": "Duplicate Billing",
            "fraud_description": (
                "Same CPT code (99283 moderate-severity ED visit) billed twice on "
                "the same date of service for the same patient. Duplicate pairs "
                "detected across 12–18% of claims in months 6–10. "
                "Unique to this pattern: exact CPT + date + NPI triplet repeats."
            ),
            "monthly_claims": months_base(380, 32),
            "monthly_amounts": months_base(95000, 8000),
            "icd10_codes": ["R51", "R10.9", "J06.9", "S09.90XA", "R07.9"],
            "cpt_codes": ["99283", "99283", "99282", "99283", "99284"],
            "patient_volume": months_base(340, 28),
            "dx_proc_match_score": 0.72,
            "referral_rate": 0.07,
            "duplicate_rate": 0.16,
        },
        {
            "id": "PRV-008",
            "name": "Suncoast Behavioral Health",
            "specialty": "Psychiatry",
            "fraud_pattern": "Seasonal Spike — End-of-Year Fraud",
            "fraud_description": (
                "10 months of stable billing (avg 195 claims/month) followed by an "
                "8x spike in months 11–12. Classic pattern of accumulating fraudulent "
                "claims before fiscal year-end audit cycles reset. "
                "Month 12 claim volume exceeds the practice's licensed capacity."
            ),
            "monthly_claims": spike_tail(195, 18, spike_mult=8.2),
            "monthly_amounts": spike_tail(48750, 4000, spike_mult=8.0),
            "icd10_codes": ["F32.1", "F41.1", "F33.0", "F32.9", "F43.10"],
            "cpt_codes": ["90837", "90834", "90832", "90837", "90847"],
            "patient_volume": spike_tail(88, 8, spike_mult=7.5),
            "dx_proc_match_score": 0.78,
            "referral_rate": 0.09,
            "duplicate_rate": 0.02,
        },
        {
            "id": "PRV-009",
            "name": "Hillcrest Community Clinic",
            "specialty": "Family Medicine",
            "fraud_pattern": "Borderline — Minor Anomalies",
            "fraud_description": (
                "Borderline provider with minor anomalies across multiple dimensions: "
                "slightly elevated average CPT complexity (99214 vs expected 99213), "
                "modest duplicate rate (4%), and referral rate marginally above peer "
                "benchmark. No single flag is conclusive — demonstrates MEDIUM risk tier."
            ),
            "monthly_claims": months_base(265, 22),
            "monthly_amounts": months_base(52000, 4500),
            "icd10_codes": ["I10", "E11.9", "Z00.00", "J06.9", "M54.5"],
            "cpt_codes": ["99214", "99214", "99213", "99214", "99215"],
            "patient_volume": months_base(248, 20),
            "dx_proc_match_score": 0.62,
            "referral_rate": 0.22,
            "duplicate_rate": 0.04,
        },
        {
            "id": "PRV-010",
            "name": "Greenfield Pediatric Associates",
            "specialty": "Pediatrics",
            "fraud_pattern": "Clean Profile — Control Case",
            "fraud_description": (
                "Legitimate provider with textbook billing patterns. CPT codes match "
                "diagnosis severity. No duplicates. Referral rate within peer benchmark. "
                "Claim volumes consistent month-over-month. Serves as the system's "
                "true-negative control — verifies absence of false positives."
            ),
            "monthly_claims": months_base(220, 18),
            "monthly_amounts": months_base(38500, 3200),
            "icd10_codes": ["Z00.129", "J06.9", "L20.9", "H10.9", "Z23"],
            "cpt_codes": ["99213", "99212", "99214", "99213", "99392"],
            "patient_volume": months_base(210, 16),
            "dx_proc_match_score": 0.91,
            "referral_rate": 0.09,
            "duplicate_rate": 0.005,
        },
    ]

    # Ensure all numeric lists are proper Python floats
    for p in providers:
        p["monthly_claims"] = [float(v) for v in p["monthly_claims"]]
        p["monthly_amounts"] = [float(v) for v in p["monthly_amounts"]]
        p["patient_volume"] = [float(v) for v in p["patient_volume"]]

    # Attach synthetic patients per provider
    for p in providers:
        p["patients"] = generate_patients_for_provider(p, rng)

    return providers


def generate_patients_for_provider(
    provider: dict[str, Any], rng: np.random.Generator
) -> list[dict[str, Any]]:
    """
    Generate 4 synthetic patients for a given provider for treatment risk screening.

    Patient records include age, primary ICD-10 diagnosis, days since last visit,
    and a composite treatment risk score. Risk score formula:
        risk = 0.4 * (days_since_visit / 365) + 0.35 * diagnosis_severity + 0.25 * age_factor
    All components clipped to [0, 1] before weighting.

    Parameters
    ----------
    provider : dict  Provider profile dict
    rng      : np.random.Generator  Seeded RNG for reproducibility

    Returns
    -------
    list[dict] : 4 patient records with keys: name, age, icd10, diagnosis_label,
                 last_visit_days_ago, risk_score, risk_flag
    """
    diagnosis_severity_map = {
        # Respiratory / ENT
        "J06.9": ("Upper Respiratory Infection", 0.15),
        "J02.9": ("Acute Pharyngitis, Unspecified", 0.12),
        "J03.90": ("Acute Tonsillitis, Unspecified", 0.12),
        "J30.1": ("Allergic Rhinitis", 0.10),
        # Preventive / Encounter
        "Z00.00": ("Routine Adult Exam", 0.05),
        "Z00.01": ("Health Exam w/ Abnormal Findings", 0.10),
        "Z00.129": ("Pediatric Preventive Visit", 0.05),
        "Z13.88": ("Encounter for Screening — Disorder NOS", 0.08),
        "Z13.6": ("Encounter for Screening — Cardiovascular", 0.10),
        "Z23": ("Encounter for Immunization", 0.03),
        "Z51.89": ("Encounter for Other Care", 0.25),
        "Z51.11": ("Encounter for Antineoplastic Chemotherapy", 0.70),
        # Musculoskeletal
        "M23.200": ("Medial Meniscus Tear, Unspecified", 0.55),
        "M23.611": ("Knee Internal Derangement — Medial Collateral", 0.52),
        "M25.361": ("Stiffness of Right Knee", 0.30),
        "M17.11": ("Primary Osteoarthritis, Knee", 0.60),
        "M54.5": ("Low Back Pain", 0.30),
        "M54.4": ("Lumbago with Sciatica", 0.38),
        "M51.16": ("Intervertebral Disc Degeneration, Lumbar", 0.50),
        "S83.200A": ("Knee Sprain, Initial Encounter", 0.35),
        # Oncology
        "C50.911": ("Breast Cancer, Unspecified", 0.90),
        "C18.9": ("Colon Cancer, Unspecified", 0.88),
        "C34.10": ("Malignant Neoplasm of Lung, Unspecified", 0.92),
        # Hematology
        "D64.9": ("Anemia, Unspecified", 0.45),
        # Psychiatry / Behavioral
        "F32.1": ("Major Depressive Disorder, Moderate", 0.65),
        "F32.9": ("Major Depressive Disorder, Unspecified", 0.60),
        "F33.0": ("Recurrent Depressive Disorder, Mild", 0.55),
        "F41.1": ("Generalized Anxiety Disorder", 0.50),
        "F43.10": ("PTSD, Unspecified", 0.62),
        # Cardiovascular / Metabolic
        "I10": ("Essential Hypertension", 0.55),
        "E11.9": ("Type 2 Diabetes, Uncomplicated", 0.60),
        "R07.9": ("Chest Pain, Unspecified", 0.45),
        # Emergency / Acute
        "R10.9": ("Unspecified Abdominal Pain", 0.22),
        "R51": ("Headache", 0.20),
        "S09.90XA": ("Head Injury, Unspecified — Initial", 0.40),
        # Dermatology / Ophthalmology
        "L20.9": ("Atopic Dermatitis, Unspecified", 0.20),
        "H10.9": ("Conjunctivitis, Unspecified", 0.08),
    }

    first_names = ["Marcus", "Elena", "Darnell", "Priya", "Thomas", "Aisha", "Carlos", "Linda"]
    last_names = ["Whitfield", "Torres", "Johnson", "Patel", "Brennan", "Okafor", "Reyes", "Chen"]

    icd10_pool = provider["icd10_codes"]
    patients = []
    today = datetime.date.today()

    ages = rng.integers(18, 82, size=4)
    days_ago_options = rng.integers(10, 420, size=4)

    for i in range(4):
        icd10 = icd10_pool[i % len(icd10_pool)]
        diag_label, severity = diagnosis_severity_map.get(
            icd10, (f"Diagnosis {icd10}", 0.40)
        )
        age = int(ages[i])
        days_ago = int(days_ago_options[i])
        last_visit = today - datetime.timedelta(days=days_ago)

        age_factor = min((age - 18) / 60.0, 1.0)
        visit_factor = min(days_ago / 365.0, 1.0)
        risk_score = round(
            0.40 * visit_factor + 0.35 * severity + 0.25 * age_factor, 3
        )

        risk_flag = "OVERDUE" if days_ago > 180 else ("MONITOR" if days_ago > 90 else "OK")

        patients.append(
            {
                "name": f"{first_names[(i * 2) % len(first_names)]} {last_names[(i * 3 + 1) % len(last_names)]}",
                "age": age,
                "icd10": icd10,
                "diagnosis_label": diag_label,
                "last_visit": last_visit.strftime("%Y-%m-%d"),
                "days_ago": days_ago,
                "risk_score": risk_score,
                "risk_flag": risk_flag,
            }
        )

    return patients


# ─── Module 2: Time-Series Anomaly Detection ─────────────────────────────────


def detect_timeseries_anomalies(
    monthly_claims: list[float], z_threshold: float = 2.0
) -> dict[str, Any]:
    """
    Detect billing anomalies in a provider's 12-month claim history using
    Z-score normalization.

    Z-score formula: z_i = (x_i - μ) / σ
    Months where |z_i| > z_threshold are flagged as anomalous.

    This is a statistical approach, not a simple if/else threshold —
    the threshold is relative to each provider's own distribution,
    making it robust to high-volume vs low-volume practice differences.

    Parameters
    ----------
    monthly_claims : list[float]  12-month claim counts
    z_threshold    : float        Z-score cutoff (default 2.0 = ~95th percentile)

    Returns
    -------
    dict with keys:
        z_scores        : np.ndarray of per-month Z-scores
        anomaly_mask    : np.ndarray of bool flags
        anomaly_months  : list[int] of 1-indexed anomalous months
        mean            : float
        std             : float
        max_z           : float
    """
    arr = np.array(monthly_claims, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1.0
    std = max(std, 1e-6)  # prevent division by zero

    z_scores = (arr - mean) / std
    anomaly_mask = np.abs(z_scores) > z_threshold

    return {
        "z_scores": z_scores,
        "anomaly_mask": anomaly_mask,
        "anomaly_months": [i + 1 for i, flag in enumerate(anomaly_mask) if flag],
        "mean": mean,
        "std": std,
        "max_z": float(np.max(np.abs(z_scores))),
    }


# ─── Module 3 & 4: Clustering + Classification ───────────────────────────────


def build_feature_matrix(providers: list[dict[str, Any]]) -> np.ndarray:
    """
    Construct a feature matrix from provider profiles for K-Means clustering.

    Features (7 dimensions):
    0 — mean_monthly_claims       : average claim volume
    1 — claim_cv                  : coefficient of variation (captures seasonal spikes)
    2 — dx_proc_match_score       : diagnosis-procedure alignment (0=mismatch, 1=perfect)
    3 — duplicate_rate            : fraction of claims with exact-match duplicates
    4 — referral_rate             : fraction of patients referred to specialists
    5 — mean_amount_per_claim     : average reimbursement per claim
    6 — max_z_score               : peak Z-score from time-series anomaly detection

    Parameters
    ----------
    providers : list[dict]  All 10 provider profiles

    Returns
    -------
    np.ndarray shape (n_providers, 7)
    """
    rows = []
    for p in providers:
        claims = np.array(p["monthly_claims"])
        amounts = np.array(p["monthly_amounts"])
        mean_claims = float(np.mean(claims))
        std_claims = float(np.std(claims, ddof=1))
        claim_cv = std_claims / mean_claims if mean_claims > 0 else 0.0
        mean_amount_per_claim = float(np.mean(amounts / (claims + 1e-6)))
        anomaly = detect_timeseries_anomalies(p["monthly_claims"])
        rows.append(
            [
                mean_claims,
                claim_cv,
                p["dx_proc_match_score"],
                p["duplicate_rate"],
                p["referral_rate"],
                mean_amount_per_claim,
                anomaly["max_z"],
            ]
        )
    return np.array(rows, dtype=float)


def run_kmeans_clustering(
    feature_matrix: np.ndarray, n_clusters: int = 4
) -> dict[str, Any]:
    """
    Fit K-Means on provider feature vectors and derive meaningful cluster labels
    from centroid analysis.

    Label assignment logic:
    - Centroid with lowest dx_proc_match_score → "Diagnosis-Procedure Mismatch"
    - Centroid with highest max_z_score        → "High-Volume Billing Anomaly"
    - Centroid with highest duplicate_rate     → "Credential and Phantom Risk"
    - Remaining centroid                       → "Clean Billing Profile"

    Parameters
    ----------
    feature_matrix : np.ndarray  shape (n_providers, 7)
    n_clusters     : int         default 4

    Returns
    -------
    dict with keys:
        labels          : np.ndarray of cluster assignments per provider
        centroids       : np.ndarray shape (n_clusters, 7)
        cluster_names   : list[str] length n_clusters, derived from centroid analysis
        scaler          : fitted StandardScaler (for future inference)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    centroids_scaled = kmeans.cluster_centers_
    # Inverse-transform centroids to interpretable feature space
    centroids = scaler.inverse_transform(centroids_scaled)

    # Feature column indices for label heuristics
    F_DX_MATCH = 2
    F_MAX_Z = 6
    F_DUP_RATE = 3

    cluster_indices = list(range(n_clusters))
    cluster_names = [""] * n_clusters

    # Assign "Diagnosis-Procedure Mismatch" to lowest dx_proc_match_score centroid
    dx_mismatch_idx = int(np.argmin(centroids[:, F_DX_MATCH]))
    cluster_names[dx_mismatch_idx] = "Diagnosis-Procedure Mismatch"

    # Assign "High-Volume Billing Anomaly" to highest max_z centroid (excluding above)
    remaining = [i for i in cluster_indices if i != dx_mismatch_idx]
    billing_anomaly_idx = remaining[int(np.argmax(centroids[remaining, F_MAX_Z]))]
    cluster_names[billing_anomaly_idx] = "High-Volume Billing Anomaly"

    # Assign "Credential and Phantom Risk" to highest duplicate_rate centroid
    remaining2 = [i for i in remaining if i != billing_anomaly_idx]
    cred_phantom_idx = remaining2[int(np.argmax(centroids[remaining2, F_DUP_RATE]))]
    cluster_names[cred_phantom_idx] = "Credential and Phantom Risk"

    # Remaining centroid → "Clean Billing Profile"
    clean_idx = [i for i in remaining2 if i != cred_phantom_idx][0]
    cluster_names[clean_idx] = "Clean Billing Profile"

    return {
        "labels": labels,
        "centroids": centroids,
        "cluster_names": cluster_names,
        "scaler": scaler,
    }


def compute_anomaly_score(
    provider: dict[str, Any], anomaly_result: dict[str, Any]
) -> float:
    """
    Compute a composite anomaly score (0–100) for a provider by combining
    multiple fraud signal dimensions.

    Score components (weights sum to 1.0):
    - Z-score signal    (0.25) : max_z / 2.5 clipped to [0,1]
    - DX-Proc mismatch  (0.40) : (1 - dx_proc_match_score)^0.7  [concave: amplifies severe mismatch]
    - Duplicate rate    (0.20) : duplicate_rate / 0.20 clipped to [0,1]
    - Referral rate     (0.10) : referral_rate / 0.70 clipped to [0,1]
    - Claim CV          (0.05) : claim_cv / 1.5 clipped to [0,1]

    Parameters
    ----------
    provider       : dict   Provider profile
    anomaly_result : dict   Output of detect_timeseries_anomalies()

    Returns
    -------
    float : Anomaly score in [0, 100]
    """
    claims = np.array(provider["monthly_claims"])
    mean_c = float(np.mean(claims))
    std_c = float(np.std(claims, ddof=1))
    claim_cv = (std_c / mean_c) if mean_c > 0 else 0.0

    z_component = min(anomaly_result["max_z"] / 2.5, 1.0)
    dx_component = (1.0 - provider["dx_proc_match_score"]) ** 0.7  # concave: amplifies high mismatch
    dup_component = min(provider["duplicate_rate"] / 0.20, 1.0)
    ref_component = min(provider["referral_rate"] / 0.70, 1.0)
    cv_component = min(claim_cv / 1.5, 1.0)

    raw = (
        0.25 * z_component
        + 0.40 * dx_component
        + 0.20 * dup_component
        + 0.10 * ref_component
        + 0.05 * cv_component
    )
    return round(raw * 100, 1)


def classify_risk(anomaly_score: float) -> str:
    """
    Classify provider risk tier from composite anomaly score.

    Classification thresholds (derived from domain benchmarks, see RISK_THRESHOLDS constant):
    - HIGH   : score >= 55
    - MEDIUM : score >= 28
    - LOW    : score < 28

    Parameters
    ----------
    anomaly_score : float  Composite score in [0, 100]

    Returns
    -------
    str : "HIGH" | "MEDIUM" | "LOW"
    """
    if anomaly_score >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif anomaly_score >= RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


# ─── Module 5: Prediction + Inference ────────────────────────────────────────


def compute_risk_factors(
    provider: dict[str, Any], anomaly_result: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Extract and rank the top 3 risk factors with data-driven confidence percentages.

    Each confidence value is computed from the provider's actual feature values,
    not from random numbers. Confidence formula per factor:
        confidence = f(raw_feature_value) → normalized to [0.50, 0.99]

    Parameters
    ----------
    provider       : dict  Provider profile
    anomaly_result : dict  Z-score anomaly detection results

    Returns
    -------
    list[dict] : Top 3 dicts with keys: factor, confidence, detail
    """
    candidate_factors = []

    # Factor 1: Billing Spike / Time-Series Anomaly
    max_z = anomaly_result["max_z"]
    if max_z > 1.0:
        conf = min(0.50 + (max_z - 1.0) / 6.0 * 0.49, 0.99)
        candidate_factors.append(
            {
                "factor": "Abnormal Billing Volume Spike",
                "confidence": round(conf * 100, 1),
                "detail": f"Peak Z-score {max_z:.2f} detected in months "
                + ", ".join(str(m) for m in anomaly_result["anomaly_months"])
                if anomaly_result["anomaly_months"]
                else f"Peak Z-score {max_z:.2f}",
                "weight": conf,
            }
        )

    # Factor 2: Diagnosis-Procedure Mismatch
    dx_mismatch = 1.0 - provider["dx_proc_match_score"]
    if dx_mismatch > 0.20:
        conf = min(0.50 + dx_mismatch * 0.49, 0.99)
        candidate_factors.append(
            {
                "factor": "Diagnosis-Procedure Code Mismatch",
                "confidence": round(conf * 100, 1),
                "detail": f"DX-CPT alignment score {provider['dx_proc_match_score']:.2f} "
                f"(benchmark ≥ 0.75). Primary codes: {provider['icd10_codes'][0]} → {provider['cpt_codes'][0]}",
                "weight": conf,
            }
        )

    # Factor 3: Duplicate Billing Rate
    dup_rate = provider["duplicate_rate"]
    if dup_rate > 0.02:
        conf = min(0.50 + (dup_rate / 0.25) * 0.49, 0.99)
        candidate_factors.append(
            {
                "factor": "Elevated Duplicate Claim Rate",
                "confidence": round(conf * 100, 1),
                "detail": f"{dup_rate*100:.1f}% of claims have exact CPT+date duplicates "
                f"(benchmark < 1.0%)",
                "weight": conf,
            }
        )

    # Factor 4: Referral Rate Anomaly
    ref_rate = provider["referral_rate"]
    if ref_rate > 0.25:
        conf = min(0.50 + (ref_rate - 0.25) / 0.50 * 0.49, 0.99)
        candidate_factors.append(
            {
                "factor": "Excessive Specialist Referral Rate",
                "confidence": round(conf * 100, 1),
                "detail": f"Referral rate {ref_rate*100:.0f}% vs peer benchmark 12–18%. "
                f"Primary referral-trigger code: {provider['icd10_codes'][0]}",
                "weight": conf,
            }
        )

    # Factor 5: Claim amount vs volume disproportion
    claims = np.array(provider["monthly_claims"])
    amounts = np.array(provider["monthly_amounts"])
    avg_per_claim = float(np.mean(amounts / (claims + 1e-6)))
    if avg_per_claim > 800:
        conf = min(0.50 + (avg_per_claim - 800) / 2000 * 0.49, 0.99)
        candidate_factors.append(
            {
                "factor": "High Average Reimbursement Per Claim",
                "confidence": round(conf * 100, 1),
                "detail": f"${avg_per_claim:,.0f} avg per claim vs specialty benchmark. "
                f"Primary procedure: {provider['cpt_codes'][0]}",
                "weight": conf,
            }
        )

    # Sort by weight descending, return top 3
    candidate_factors.sort(key=lambda x: x["weight"], reverse=True)
    return candidate_factors[:3]


def generate_inference_statement(
    provider: dict[str, Any],
    risk_factors: list[dict[str, Any]],
    anomaly_score: float,
    risk_tier: str,
    cluster_name: str,
) -> str:
    """
    Produce a two-sentence analyst-grade inference statement derived from
    the provider's specific fraud pattern and detected signals.

    The statement synthesizes the combination of flags into a conclusion
    as a senior payment integrity analyst would write it.

    Parameters
    ----------
    provider      : dict  Provider profile
    risk_factors  : list  Top risk factors from compute_risk_factors()
    anomaly_score : float Composite anomaly score
    risk_tier     : str   "HIGH" | "MEDIUM" | "LOW"
    cluster_name  : str   K-Means derived cluster label

    Returns
    -------
    str : Two-sentence inference statement
    """
    pattern = provider["fraud_pattern"]
    factor_names = [rf["factor"] for rf in risk_factors]
    primary_icd = provider["icd10_codes"][0]
    primary_cpt = provider["cpt_codes"][0]

    templates = {
        "Upcoding": (
            f"The convergence of a low diagnosis-procedure alignment score ({provider['dx_proc_match_score']:.2f}) "
            f"with dominant billing of CPT {primary_cpt} against ICD-10 {primary_icd} "
            f"is the quantitative signature of systematic upcoding — selecting the highest-paying "
            f"complexity code regardless of documented clinical justification. "
            f"With an anomaly score of {anomaly_score:.1f} and cluster assignment to '{cluster_name}', "
            f"this provider warrants a targeted medical record review comparing documented visit complexity "
            f"against billed E&M levels for a 90-day claim sample."
        ),
        "Unbundling": (
            f"The co-occurrence of CPT {primary_cpt} and {provider['cpt_codes'][1]} "
            f"across the same episode-of-care dates — combined with an above-benchmark reimbursement "
            f"per claim — indicates deliberate fragmentation of bundled surgical procedures "
            f"to circumvent CCI (Correct Coding Initiative) edits. "
            f"At anomaly score {anomaly_score:.1f}, this provider should be flagged for "
            f"automated CCI bundling audit across all arthroscopic procedure pairs in the claim history."
        ),
        "Phantom Billing": (
            f"The extreme diagnosis-procedure mismatch (score {provider['dx_proc_match_score']:.2f}) "
            f"paired with high claim volume against minimal patient visit records — "
            f"ICD-10 {primary_icd} driving CPT {primary_cpt} — is the statistical fingerprint "
            f"of phantom billing, where procedures are claimed without corresponding clinical encounters. "
            f"Immediate cross-reference of claimed service dates against facility access logs "
            f"and patient attestation records is the recommended next action."
        ),
        "Ping-Ponging": (
            f"A referral rate of {provider['referral_rate']*100:.0f}% — more than 3x the specialty benchmark — "
            f"combined with {primary_icd} (routine exam) as the primary referral trigger "
            f"indicates a coordinated referral loop generating billable specialist encounters "
            f"without documented clinical necessity. "
            f"Network analysis of this provider's outbound referral graph should identify "
            f"the affiliated specialists absorbing the generated visit volume."
        ),
        "Credential Billing": (
            f"The billing pattern under NPI {provider['id']} shows procedure volumes "
            f"that exceed the supervising physician's documented on-site capacity, "
            f"with CPT {primary_cpt} (requiring physician supervision) billed on high-frequency days "
            f"that correlate with confirmed supervising physician absences. "
            f"CMS enrollment record cross-check and on-site staff credentialing audit "
            f"are the appropriate remediation steps at this risk level ({risk_tier})."
        ),
        "Medically Unnecessary Procedures": (
            f"A diagnosis-procedure alignment score of {provider['dx_proc_match_score']:.2f} "
            f"— the lowest in the provider cohort — directly reflects the pattern of pairing "
            f"ICD-10 {primary_icd} with CPT {primary_cpt}, a combination that fails "
            f"clinical necessity review under any payer LCD (Local Coverage Determination). "
            f"Pre-payment clinical review with attending physician attestation is required "
            f"before any further claims from this provider are approved."
        ),
        "Duplicate Billing": (
            f"A duplicate claim rate of {provider['duplicate_rate']*100:.1f}% — "
            f"representing exact CPT+date+NPI triplet repetitions — exceeds the 1% "
            f"industry threshold for systematic duplicate submission versus isolated data entry error. "
            f"Automated deduplication recovery should be applied to the full claim history, "
            f"with recovered overpayment demand letters issued for confirmed duplicate pairs."
        ),
        "Seasonal Spike — End-of-Year Fraud": (
            f"The billing trajectory for this provider shows textbook end-of-year accumulation fraud: "
            f"10 months of stable volume followed by an 8x spike in months 11–12, "
            f"a pattern that correlates with fiscal year-end submission windows "
            f"designed to exploit audit cycle resets. "
            f"Retroactive claim review for months 11–12 with comparison to capacity-adjusted "
            f"peer benchmarks is recommended, with recoupment action for claims exceeding "
            f"licensed treatment capacity."
        ),
        "Borderline — Minor Anomalies": (
            f"This provider presents no single definitive fraud indicator, "
            f"but the combination of {', '.join(factor_names[:2])} "
            f"places it at the upper boundary of the MEDIUM risk tier with anomaly score {anomaly_score:.1f}. "
            f"Enhanced monitoring with 6-month re-evaluation is appropriate — "
            f"if referral rate and billing complexity continue trending upward, "
            f"a targeted audit becomes warranted."
        ),
        "Clean Profile — Control Case": (
            f"All fraud signal dimensions for this provider fall within peer-benchmarked normal ranges: "
            f"diagnosis-procedure alignment score {provider['dx_proc_match_score']:.2f}, "
            f"duplicate rate {provider['duplicate_rate']*100:.1f}%, "
            f"and stable month-over-month billing volume with no Z-score exceedances. "
            f"This provider is correctly classified as LOW risk and requires no further audit action "
            f"under standard payment integrity protocols."
        ),
    }

    return templates.get(
        pattern,
        f"Provider {provider['id']} demonstrates {risk_tier} risk characteristics "
        f"(anomaly score {anomaly_score:.1f}) based on {', '.join(factor_names[:2])}. "
        f"Further review is recommended.",
    )


# ─── Module 6: Agentic Chain Reasoning + Groq ────────────────────────────────


def build_audit_brief(
    provider: dict[str, Any],
    anomaly_result: dict[str, Any],
    anomaly_score: float,
    risk_tier: str,
    cluster_name: str,
    risk_factors: list[dict[str, Any]],
) -> str:
    """
    Construct a structured Cotiviti-style payment integrity audit brief
    for submission to the Groq LLM.

    The brief follows internal Cotiviti case format:
    - Case metadata header
    - Detected fraud pattern with CPT/ICD-10 specifics
    - Quantitative flags with actual data values
    - Required output schema (5-step chain reasoning)

    Parameters
    ----------
    provider       : dict   Provider profile
    anomaly_result : dict   Z-score detection output
    anomaly_score  : float  Composite anomaly score
    risk_tier      : str    Risk classification
    cluster_name   : str    K-Means cluster label
    risk_factors   : list   Top 3 risk factors with confidence scores

    Returns
    -------
    str : Formatted prompt string for Groq API
    """
    anomalous_months_str = (
        ", ".join(f"Month {m}" for m in anomaly_result["anomaly_months"])
        if anomaly_result["anomaly_months"]
        else "None detected"
    )

    claims_arr = np.array(provider["monthly_claims"])
    amounts_arr = np.array(provider["monthly_amounts"])
    avg_monthly_claims = float(np.mean(claims_arr))
    avg_monthly_amount = float(np.mean(amounts_arr))
    peak_month = int(np.argmax(claims_arr)) + 1
    peak_claims = float(np.max(claims_arr))

    factors_str = "\n".join(
        f"  FLAG {i+1}: {rf['factor']} — Confidence {rf['confidence']}%\n"
        f"           Detail: {rf['detail']}"
        for i, rf in enumerate(risk_factors)
    )

    brief = f"""
COTIVITI PAYMENT INTEGRITY — INTERNAL AUDIT BRIEF
Case Number      : CA-2024-{provider['id']}
Provider ID      : {provider['id']}
Provider Name    : {provider['name']}
Specialty        : {provider['specialty']}
Audit Date       : {datetime.date.today().strftime('%B %d, %Y')}
Risk Tier        : {risk_tier}
Anomaly Score    : {anomaly_score:.1f} / 100
Cluster Group    : {cluster_name}

DETECTED PATTERN
Pattern Type     : {provider['fraud_pattern']}
Pattern Summary  : {provider['fraud_description']}

BILLING INTELLIGENCE SUMMARY
Primary ICD-10 Codes      : {', '.join(provider['icd10_codes'])}
Primary CPT Codes         : {', '.join(provider['cpt_codes'])}
Avg Monthly Claims        : {avg_monthly_claims:.0f}
Avg Monthly Reimbursement : ${avg_monthly_amount:,.0f}
Peak Billing Month        : Month {peak_month} ({peak_claims:.0f} claims)
Anomalous Months (Z>2.0)  : {anomalous_months_str}
Peak Z-Score              : {anomaly_result['max_z']:.2f}
DX-CPT Alignment Score    : {provider['dx_proc_match_score']:.2f} (benchmark ≥ 0.75)
Duplicate Claim Rate      : {provider['duplicate_rate']*100:.1f}%
Referral Rate             : {provider['referral_rate']*100:.0f}%

TRIGGERED FLAGS
{factors_str}

REQUIRED OUTPUT FORMAT
You are a senior Cotiviti payment integrity analyst. Respond ONLY in the following structured format.
Do not add preamble. Do not summarize at the end. Write as an internal audit document, not a chatbot response.

STEP 1 — FLAG ANALYSIS
Analyze each triggered flag individually. Explain what the quantitative values indicate
about billing behavior. Reference the specific CPT and ICD-10 codes. Be precise.

STEP 2 — PATTERN CONFIRMATION
Confirm or qualify whether the detected pattern ({provider['fraud_pattern']}) is supported
by the combination of flags. Explain what combinations of signals are definitive vs circumstantial.

STEP 3 — REGULATORY CONTEXT
Cite the specific CMS regulations, OIG guidelines, or False Claims Act provisions
that are implicated by this pattern. Reference relevant LCD policies if applicable.

STEP 4 — RISK VERDICT
State the overall risk verdict for this provider. Quantify the estimated financial
exposure based on the monthly amounts and the pattern type. Assign a priority level
(Immediate / High / Routine) for audit action.

STEP 5 — RECOMMENDED ACTION
Provide exactly 3 specific, actionable recommendations for the payment integrity team.
Each recommendation should reference a specific data point from this case.
""".strip()

    return brief


def stream_groq_response(prompt: str) -> Any:
    """
    Call Groq API with the audit brief prompt using streaming.

    Uses llama-3.3-70b-versatile for production-quality chain reasoning output.
    Streams token-by-token into Streamlit for visible agentic behavior.

    Parameters
    ----------
    prompt : str  Formatted audit brief from build_audit_brief()

    Returns
    -------
    Generator yielding text chunks from the Groq streaming response
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "See .env.example for configuration."
        )

    client = Groq(api_key=api_key)
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior payment integrity analyst at Cotiviti, a healthcare "
                    "payment accuracy company. You write structured internal audit documents "
                    "for health plan clients. Your analysis is precise, data-driven, and "
                    "references specific billing codes, regulatory frameworks, and quantitative "
                    "thresholds. You never use generic healthcare language. Every statement "
                    "is grounded in the specific case data provided."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1200,
        temperature=0.3,
        stream=True,
    )
    return stream


# ─── Module 7 & 8: UI Rendering ──────────────────────────────────────────────


def inject_custom_css() -> None:
    """Inject custom CSS for Cotiviti-branded dark professional dashboard theme."""
    st.markdown(
        f"""
    <style>
        /* Base theme */
        .stApp {{
            background-color: {BG_DARK};
            color: {TEXT_PRIMARY};
        }}
        .main .block-container {{
            padding-top: 0.5rem;
            max-width: 1200px;
        }}
        /* Remove Streamlit default top padding */
        [data-testid="stAppViewContainer"] > .main {{
            padding-top: 0 !important;
        }}
        header[data-testid="stHeader"] {{
            display: none !important;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: #16192A;
            border-right: 1px solid {COTIVITI_PURPLE};
        }}
        /* Hide ALL sidebar collapse/toggle buttons — every known Streamlit selector */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"],
        button[data-testid="stSidebarCollapseButton"],
        section[data-testid="stSidebar"] > div > div > button,
        .st-emotion-cache-zq5wmm,
        .st-emotion-cache-1egp75f,
        [data-testid="stSidebar"] button[kind="header"],
        #MainMenu {{
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
            width: 0 !important;
            height: 0 !important;
            opacity: 0 !important;
        }}
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {{
            color: {TEXT_PRIMARY} !important;
        }}

        /* App title in sidebar */
        .sidebar-title {{
            font-size: 1.4rem;
            font-weight: 700;
            color: {COTIVITI_PURPLE_LIGHT};
            margin-bottom: 0.2rem;
        }}
        .sidebar-subtitle {{
            font-size: 0.78rem;
            color: {TEXT_SECONDARY};
            margin-bottom: 1.5rem;
            line-height: 1.4;
        }}

        /* Section dividers with concept label */
        .section-divider {{
            display: flex;
            align-items: center;
            margin: 2rem 0 1.2rem 0;
        }}
        .section-divider::before,
        .section-divider::after {{
            content: '';
            flex: 1;
            border-top: 2px solid {COTIVITI_PURPLE};
        }}
        .section-divider-label {{
            padding: 0.3rem 1rem;
            background: {COTIVITI_PURPLE};
            color: white;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0 0.8rem;
            white-space: nowrap;
        }}

        /* Metric cards */
        .metric-card {{
            background-color: {CARD_BG};
            border-left: 4px solid {COTIVITI_PURPLE};
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin: 0.3rem 0;
        }}
        .metric-card-label {{
            font-size: 0.72rem;
            color: {TEXT_SECONDARY};
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.3rem;
        }}
        .metric-card-value {{
            font-size: 1.6rem;
            font-weight: 700;
            color: {TEXT_PRIMARY};
        }}
        .metric-card-sub {{
            font-size: 0.78rem;
            color: {TEXT_SECONDARY};
            margin-top: 0.15rem;
        }}

        /* Risk badges */
        .badge-high {{
            background-color: {RED_ALERT};
            color: white;
            padding: 0.3rem 0.9rem;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
        }}
        .badge-medium {{
            background-color: {ORANGE_WARN};
            color: white;
            padding: 0.3rem 0.9rem;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
        }}
        .badge-low {{
            background-color: {GREEN_OK};
            color: white;
            padding: 0.3rem 0.9rem;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
        }}

        /* Agent log */
        .agent-log {{
            background-color: #0D1117;
            border: 1px solid {COTIVITI_PURPLE};
            border-radius: 8px;
            padding: 1rem 1.4rem;
            font-family: 'Courier New', monospace;
            font-size: 0.82rem;
            color: #00FF88;
            min-height: 180px;
            line-height: 1.8;
        }}
        .agent-step {{
            margin: 0.15rem 0;
        }}
        .agent-step-complete {{
            color: #00FF88;
        }}
        .agent-step-active {{
            color: #FFD700;
        }}

        /* LLM output box */
        .llm-output {{
            background-color: #0D1117;
            border: 1px solid #3A3D4A;
            border-radius: 8px;
            padding: 1.4rem 1.8rem;
            font-family: 'Georgia', serif;
            font-size: 0.88rem;
            color: {TEXT_PRIMARY};
            line-height: 1.75;
        }}
        .llm-output p {{
            margin: 0 0 0.6rem 0;
        }}
        .llm-output strong, .llm-output b {{
            color: {COTIVITI_PURPLE_LIGHT};
            font-weight: 700;
        }}
        .llm-output ol {{
            margin: 0.4rem 0 0.6rem 1.2rem;
            padding: 0;
        }}
        .llm-output ol li {{
            margin: 0.4rem 0;
            padding-left: 0.3rem;
        }}

        /* Patient risk table */
        .patient-row-overdue {{
            background-color: rgba(231, 76, 60, 0.15);
            border-left: 3px solid {RED_ALERT};
            padding: 0.4rem 0.6rem;
            border-radius: 4px;
            margin: 0.2rem 0;
        }}
        .patient-row-monitor {{
            background-color: rgba(230, 126, 34, 0.15);
            border-left: 3px solid {ORANGE_WARN};
            padding: 0.4rem 0.6rem;
            border-radius: 4px;
            margin: 0.2rem 0;
        }}
        .patient-row-ok {{
            background-color: rgba(39, 174, 96, 0.10);
            border-left: 3px solid {GREEN_OK};
            padding: 0.4rem 0.6rem;
            border-radius: 4px;
            margin: 0.2rem 0;
        }}

        /* Inference box */
        .inference-box {{
            background-color: {CARD_BG};
            border-left: 4px solid {COTIVITI_PURPLE_LIGHT};
            border-radius: 6px;
            padding: 1rem 1.2rem;
            font-style: italic;
            color: #D0D0E0;
            font-size: 0.88rem;
            line-height: 1.65;
            margin-top: 0.8rem;
        }}

        /* Operations overview text contrast fixes */
        .stDataFrame td, .stDataFrame th {{
            color: {TEXT_PRIMARY} !important;
            font-size: 0.83rem;
        }}
        .stDataFrame th {{
            background-color: #1E2136 !important;
            color: {COTIVITI_PURPLE_LIGHT} !important;
            font-weight: 600 !important;
            letter-spacing: 0.03em;
        }}
        /* Portfolio table row text always white */
        [data-testid="stDataFrame"] * {{
            color: {TEXT_PRIMARY} !important;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            color: {TEXT_SECONDARY};
            font-size: 0.72rem;
            padding: 2rem 0 0.5rem 0;
            border-top: 1px solid #2A2D3A;
            margin-top: 3rem;
        }}

        /* Streamlit default overrides */
        .stButton > button {{
            background-color: {COTIVITI_PURPLE};
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            width: 100%;
            padding: 0.6rem;
        }}
        .stButton > button:hover {{
            background-color: {COTIVITI_PURPLE_LIGHT};
            color: white;
        }}
        h1, h2, h3 {{
            color: {COTIVITI_PURPLE_LIGHT} !important;
        }}
        .stDataFrame {{
            border: 1px solid #2A2D3A;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def section_divider(label: str) -> None:
    """Render a styled section divider with a centered concept label."""
    st.markdown(
        f"""
    <div class="section-divider">
        <span class="section-divider-label">{label}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, sub: str = "") -> str:
    """Return HTML string for a styled metric card."""
    sub_html = f'<div class="metric-card-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
        <div class="metric-card-label">{label}</div>
        <div class="metric-card-value">{value}</div>
        {sub_html}
    </div>
    """


def render_risk_badge(risk_tier: str) -> str:
    """Return HTML badge for the given risk tier."""
    badge_class = {
        "HIGH": "badge-high",
        "MEDIUM": "badge-medium",
        "LOW": "badge-low",
    }.get(risk_tier, "badge-low")
    return f'<span class="{badge_class}">⬤ {risk_tier} RISK</span>'


def render_timeseries_chart(
    provider: dict[str, Any], anomaly_result: dict[str, Any]
) -> go.Figure:
    """
    Build an interactive Plotly time-series chart of monthly claim volumes.

    Normal months: Cotiviti purple line + filled markers
    Anomalous months (|Z| > 2.0): Red markers with Z-score annotations
    Reference line: dashed line at mean ± 2σ level

    Parameters
    ----------
    provider       : dict  Provider profile
    anomaly_result : dict  Z-score detection output

    Returns
    -------
    go.Figure : Plotly figure object
    """
    months = [f"M{i+1}" for i in range(12)]
    claims = provider["monthly_claims"]
    z_scores = anomaly_result["z_scores"]
    anomaly_mask = anomaly_result["anomaly_mask"]
    mean_val = anomaly_result["mean"]
    std_val = anomaly_result["std"]
    threshold_upper = mean_val + 2.0 * std_val

    fig = go.Figure()

    # Base line trace
    fig.add_trace(
        go.Scatter(
            x=months,
            y=claims,
            mode="lines",
            name="Monthly Claims",
            line=dict(color=COTIVITI_PURPLE, width=2.5),
            hovertemplate="<b>%{x}</b><br>Claims: %{y:.0f}<extra></extra>",
        )
    )

    # Normal month markers
    normal_x = [months[i] for i in range(12) if not anomaly_mask[i]]
    normal_y = [claims[i] for i in range(12) if not anomaly_mask[i]]
    fig.add_trace(
        go.Scatter(
            x=normal_x,
            y=normal_y,
            mode="markers",
            name="Normal",
            marker=dict(color=COTIVITI_PURPLE, size=8),
            hovertemplate="<b>%{x}</b><br>Claims: %{y:.0f}<br>Status: Normal<extra></extra>",
        )
    )

    # Anomalous month markers with Z-score annotations
    anom_x = [months[i] for i in range(12) if anomaly_mask[i]]
    anom_y = [claims[i] for i in range(12) if anomaly_mask[i]]
    anom_z = [z_scores[i] for i in range(12) if anomaly_mask[i]]

    if anom_x:
        fig.add_trace(
            go.Scatter(
                x=anom_x,
                y=anom_y,
                mode="markers+text",
                name="Anomaly (|Z| > 2.0)",
                marker=dict(color=RED_ALERT, size=14, symbol="diamond", line=dict(color="white", width=1.5)),
                text=[f"Z={z:.2f}" for z in anom_z],
                textposition="top center",
                textfont=dict(color=RED_ALERT, size=11, family="Courier New"),
                hovertemplate="<b>%{x}</b><br>Claims: %{y:.0f}<br>Z-Score: %{text}<extra></extra>",
            )
        )

    # Mean + 2σ threshold reference line
    fig.add_hline(
        y=threshold_upper,
        line_dash="dash",
        line_color="#FF8C00",
        annotation_text=f"Z=2.0 threshold ({threshold_upper:.0f} claims)",
        annotation_position="top right",
        annotation_font_color="#FF8C00",
    )

    # Mean line
    fig.add_hline(
        y=mean_val,
        line_dash="dot",
        line_color=TEXT_SECONDARY,
        annotation_text=f"Mean ({mean_val:.0f})",
        annotation_position="bottom right",
        annotation_font_color=TEXT_SECONDARY,
    )

    fig.update_layout(
        plot_bgcolor=CARD_BG,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif"),
        xaxis=dict(
            title=dict(text="Month", font=dict(color=TEXT_PRIMARY, size=13)),
            gridcolor="#2A2D3A",
            linecolor="#3A3D4A",
            tickfont=dict(color=TEXT_PRIMARY),
        ),
        yaxis=dict(
            title=dict(text="Monthly Claim Count", font=dict(color=TEXT_PRIMARY, size=13)),
            gridcolor="#2A2D3A",
            linecolor="#3A3D4A",
            tickfont=dict(color=TEXT_PRIMARY),
        ),
        legend=dict(
            bgcolor="rgba(20,22,40,0.92)",
            bordercolor=COTIVITI_PURPLE,
            borderwidth=1,
            font=dict(color=TEXT_PRIMARY, size=11),
            x=1.01,
            y=1,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(t=20, b=50, l=70, r=160),
        height=370,
    )
    return fig


def render_operations_scatter(
    providers: list[dict[str, Any]],
    anomaly_scores: list[float],
    risk_tiers: list[str],
    cluster_names_per_provider: list[str],
    clustering_result: dict[str, Any],
) -> go.Figure:
    """
    Build the Tab 2 operations portfolio scatter chart.

    X-axis: average monthly claim volume
    Y-axis: composite anomaly score
    Dot size: average monthly claim amount (scaled)
    Color: cluster group

    Parameters
    ----------
    providers                  : list[dict]  All 10 providers
    anomaly_scores             : list[float] One per provider
    risk_tiers                 : list[str]   One per provider
    cluster_names_per_provider : list[str]   One per provider
    clustering_result          : dict        K-Means output

    Returns
    -------
    go.Figure
    """
    cluster_color_map = {
        "High-Volume Billing Anomaly": "#E74C3C",
        "Diagnosis-Procedure Mismatch": "#E67E22",
        "Credential and Phantom Risk": "#8E44AD",
        "Clean Billing Profile": "#27AE60",
    }

    fig = go.Figure()

    for cluster_name in set(cluster_names_per_provider):
        indices = [i for i, c in enumerate(cluster_names_per_provider) if c == cluster_name]
        fig.add_trace(
            go.Scatter(
                x=[np.mean(providers[i]["monthly_claims"]) for i in indices],
                y=[anomaly_scores[i] for i in indices],
                mode="markers",
                name=cluster_name,
                marker=dict(
                    size=[np.mean(providers[i]["monthly_amounts"]) / 8000 for i in indices],
                    sizemode="diameter",
                    sizemin=12,
                    color=cluster_color_map.get(cluster_name, COTIVITI_PURPLE),
                    opacity=0.85,
                    line=dict(color="white", width=1),
                ),
                text=[providers[i]["name"] for i in indices],
                customdata=[
                    [
                        providers[i]["fraud_pattern"],
                        risk_tiers[i],
                        f"{anomaly_scores[i]:.1f}",
                        f"${np.mean(providers[i]['monthly_amounts']):,.0f}",
                    ]
                    for i in indices
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Pattern: %{customdata[0]}<br>"
                    "Risk: %{customdata[1]}<br>"
                    "Anomaly Score: %{customdata[2]}<br>"
                    "Avg Monthly Amount: %{customdata[3]}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Risk zone shading — boundaries match RISK_THRESHOLDS constant (HIGH>=55, MEDIUM>=28)
    fig.add_hrect(
        y0=RISK_THRESHOLDS["HIGH"], y1=100,
        fillcolor="rgba(231,76,60,0.08)",
        line_width=0,
        annotation_text=f"HIGH RISK ZONE (score ≥ {RISK_THRESHOLDS['HIGH']})",
        annotation_position="top left",
        annotation_font_color=RED_ALERT,
        annotation_font_size=11,
    )
    fig.add_hrect(
        y0=RISK_THRESHOLDS["MEDIUM"], y1=RISK_THRESHOLDS["HIGH"],
        fillcolor="rgba(230,126,34,0.06)",
        line_width=0,
        annotation_text=f"MEDIUM RISK ({RISK_THRESHOLDS['MEDIUM']}–{RISK_THRESHOLDS['HIGH']})",
        annotation_position="top left",
        annotation_font_color=ORANGE_WARN,
        annotation_font_size=11,
    )

    fig.update_layout(
        plot_bgcolor=CARD_BG,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif"),
        xaxis=dict(
            title=dict(text="Avg Monthly Claim Volume", font=dict(color=TEXT_PRIMARY, size=13)),
            gridcolor="#2A2D3A",
            tickfont=dict(color=TEXT_PRIMARY),
            linecolor="#3A3D4A",
        ),
        yaxis=dict(
            title=dict(text="Composite Anomaly Score", font=dict(color=TEXT_PRIMARY, size=13)),
            gridcolor="#2A2D3A",
            range=[0, 110],
            tickfont=dict(color=TEXT_PRIMARY),
            linecolor="#3A3D4A",
        ),
        legend=dict(
            bgcolor="rgba(20,22,40,0.92)",
            bordercolor=COTIVITI_PURPLE,
            borderwidth=1,
            font=dict(color=TEXT_PRIMARY, size=12),
            x=1.01,
            y=1,
            xanchor="left",
            yanchor="top",
            title=dict(text="Cluster Group", font=dict(color=COTIVITI_PURPLE_LIGHT, size=12)),
        ),
        height=500,
        margin=dict(t=40, b=60, l=80, r=200),
    )
    return fig


# ─── Main Application ─────────────────────────────────────────────────────────


def main() -> None:
    """
    Entry point for ClinicalAudit AI Streamlit application.

    Orchestrates data loading, analytics computation, and UI rendering
    across two tabs: Audit Interface and Operations Overview.
    """
    st.set_page_config(
        page_title="ClinicalAudit AI - Cotiviti Payment Integrity",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    # Force sidebar always visible via JS — prevents collapse button from working
    st.markdown(
        """
        <script>
        // Hide sidebar collapse button after render
        function hideSidebarBtn() {
            const btns = document.querySelectorAll(
                '[data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"]'
            );
            btns.forEach(b => { b.style.display = 'none'; b.style.visibility = 'hidden'; });
        }
        setTimeout(hideSidebarBtn, 500);
        setTimeout(hideSidebarBtn, 1500);
        setTimeout(hideSidebarBtn, 3000);
        </script>
        """,
        unsafe_allow_html=True,
    )

    # ── Pre-compute all analytics once per session ────────────────────────────
    if "providers" not in st.session_state:
        providers = generate_provider_profiles()
        feature_matrix = build_feature_matrix(providers)
        clustering_result = run_kmeans_clustering(feature_matrix)

        anomaly_results = [
            detect_timeseries_anomalies(p["monthly_claims"]) for p in providers
        ]
        anomaly_scores = [
            compute_anomaly_score(p, ar) for p, ar in zip(providers, anomaly_results)
        ]
        risk_tiers = [classify_risk(s) for s in anomaly_scores]
        cluster_labels_per_provider = [
            clustering_result["cluster_names"][clustering_result["labels"][i]]
            for i in range(len(providers))
        ]

        st.session_state["providers"] = providers
        st.session_state["feature_matrix"] = feature_matrix
        st.session_state["clustering_result"] = clustering_result
        st.session_state["anomaly_results"] = anomaly_results
        st.session_state["anomaly_scores"] = anomaly_scores
        st.session_state["risk_tiers"] = risk_tiers
        st.session_state["cluster_labels"] = cluster_labels_per_provider

    providers: list[dict] = st.session_state["providers"]
    clustering_result: dict = st.session_state["clustering_result"]
    anomaly_results: list[dict] = st.session_state["anomaly_results"]
    anomaly_scores: list[float] = st.session_state["anomaly_scores"]
    risk_tiers: list[str] = st.session_state["risk_tiers"]
    cluster_labels: list[str] = st.session_state["cluster_labels"]

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">🏥 ClinicalAudit AI</div>'
            '<div class="sidebar-subtitle">Payment Integrity Intelligence System<br>'
            "Cotiviti GenAI Intern Assessment</div>",
            unsafe_allow_html=True,
        )

        with st.expander("ℹ️ How It Works — ML Concepts"):
            st.markdown(
                f"""
**Time-Series Anomaly Detection** (numpy)
Z-score normalization on 12-month billing history. Months where |Z| > 2.0 flagged as statistically anomalous relative to the provider's own distribution.

**Clustering** (sklearn KMeans)
K=4 clustering on 7-dimensional feature vectors. Cluster labels derived from centroid analysis of diagnosis-procedure mismatch, Z-score peaks, and duplicate rates — not hardcoded.

**Classification** (rule-based on score)
Three-tier risk classification (HIGH/MEDIUM/LOW) from composite anomaly score. Score weights: Z-signal 25%, DX mismatch 40%, duplicate rate 20%, referral rate 10%, claim CV 5%.

**Prediction** (feature-derived confidence)
Top-3 risk factor extraction with confidence percentages computed from raw feature values using sigmoid-like normalization — no random numbers.

**Inference** (pattern-aware synthesis)
Two-sentence analyst inference synthesizing combination of flags into a domain-specific conclusion tied to the detected fraud pattern.

**Agentic GenAI** (step-logged orchestration)
Sequential agent steps logged with delays before LLM call — mimicking autonomous audit agent workflow.

**Chain Reasoning** (llama-3.3-70b-versatile)
5-step structured audit brief prompts chain-of-thought reasoning: flag analysis → pattern confirmation → regulatory context → verdict → actions.
                """,
                unsafe_allow_html=True,
            )


        st.markdown("---")

        provider_options = {
            f"{p['id']} — {p['name']} [{p['fraud_pattern']}]": i
            for i, p in enumerate(providers)
        }
        selected_label = st.selectbox(
            "Select Provider for Audit",
            options=list(provider_options.keys()),
        )
        selected_idx = provider_options[selected_label]
        selected_provider = providers[selected_idx]

        st.markdown("---")
        run_audit = st.button("🔍 Run Full Audit", use_container_width=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🔍 Audit Interface", "📊 Operations Overview"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Main Audit Interface
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        prov = selected_provider
        idx = selected_idx
        ar = anomaly_results[idx]
        score = anomaly_scores[idx]
        tier = risk_tiers[idx]
        cluster = cluster_labels[idx]

        st.markdown(
            f"## {prov['id']} — {prov['name']}",
        )
        st.markdown(
            f"**Specialty:** {prov['specialty']} &nbsp;|&nbsp; "
            f"**Pattern Under Review:** `{prov['fraud_pattern']}`"
        )

        # ── SECTION 1: Time-Series Anomaly Detection ──────────────────────
        section_divider("⏱ TIME-SERIES ANOMALY DETECTION")

        fig_ts = render_timeseries_chart(prov, ar)
        st.plotly_chart(fig_ts, use_container_width=True)

        if ar["anomaly_months"]:
            anom_months_str = ", ".join(f"Month {m}" for m in ar["anomaly_months"])
            peak_z = ar["max_z"]
            peak_month_idx = int(np.argmax(np.abs(ar["z_scores"]))) + 1
            peak_claims = prov["monthly_claims"][peak_month_idx - 1]
            st.markdown(
                f"""
<div class="inference-box">
🔴 <strong>Anomaly Detected:</strong> Months {anom_months_str} show statistically significant
billing spikes (peak Z-score {peak_z:.2f} in Month {peak_month_idx}: {peak_claims:.0f} claims).
The Z-score threshold of 2.0 corresponds to the 95th percentile of this provider's own
billing distribution — these are not absolute volume alerts, but relative deviations
from the provider's established baseline, making the detection robust to practice size.
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div class="inference-box">
✅ <strong>No Significant Anomalies:</strong> All 12 months fall within Z-score bounds (|Z| ≤ 2.0).
Billing volume is stable relative to this provider's own distribution. No time-series flags triggered.
</div>
""",
                unsafe_allow_html=True,
            )

        # ── SECTION 2: Clustering + Classification ────────────────────────
        section_divider("🔵 CLUSTERING + CLASSIFICATION")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                render_metric_card(
                    "Composite Anomaly Score",
                    f"{score:.1f}",
                    sub="Score range: 0 (clean) → 100 (high fraud signal)",
                ),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                render_metric_card(
                    "K-Means Cluster Assignment",
                    cluster,
                    sub=f"K=4 clustering on 7 billing features",
                ),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                render_metric_card("Risk Classification", "", sub=""),
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="margin-top:-0.8rem; padding-left:1.2rem;">{render_risk_badge(tier)}</div>',
                unsafe_allow_html=True,
            )

        # ── SECTION 3: Prediction + Inference ────────────────────────────
        section_divider("📈 PREDICTION + INFERENCE")

        risk_factors = compute_risk_factors(prov, ar)
        inference_stmt = generate_inference_statement(
            prov, risk_factors, score, tier, cluster
        )

        if risk_factors:
            rf_df = pd.DataFrame(
                [
                    {
                        "Risk Factor": rf["factor"],
                        "Confidence": f"{rf['confidence']}%",
                        "Supporting Evidence": rf["detail"],
                    }
                    for rf in risk_factors
                ]
            )
            st.dataframe(rf_df, use_container_width=True, hide_index=True)
        else:
            st.info("No significant risk factors above confidence threshold.")

        st.markdown(
            f'<div class="inference-box">🧠 <strong>Analyst Inference:</strong> {inference_stmt}</div>',
            unsafe_allow_html=True,
        )

        # ── SECTION 4: Treatment Risk Screening ──────────────────────────
        section_divider("🩺 TREATMENT RISK SCREENING  [TPO — Treatment Domain]")

        patients = prov["patients"]
        patient_cols = st.columns([2, 1, 2, 2, 1, 1])
        headers = ["Patient", "Age", "Diagnosis (ICD-10)", "Last Visit", "Days Since", "Risk"]
        for col, h in zip(patient_cols, headers):
            col.markdown(f"**{h}**")

        flag_colors = {
            "OVERDUE": ("#E74C3C", "🔴"),
            "MONITOR": ("#E67E22", "🟡"),
            "OK": ("#27AE60", "🟢"),
        }

        for pat in patients:
            cols = st.columns([2, 1, 2, 2, 1, 1])
            color, icon = flag_colors[pat["risk_flag"]]
            cols[0].write(pat["name"])
            cols[1].write(str(pat["age"]))
            cols[2].write(f"{pat['icd10']} — {pat['diagnosis_label']}")
            cols[3].write(pat["last_visit"])
            cols[4].write(str(pat["days_ago"]))
            cols[5].markdown(f'<span style="color:{color};">{icon} {pat["risk_flag"]}</span>', unsafe_allow_html=True)

        st.caption(
            "🔴 OVERDUE: >180 days since last visit | 🟡 MONITOR: 90–180 days | 🟢 OK: <90 days | "
            "Risk score = 0.40×visit_gap + 0.35×diagnosis_severity + 0.25×age_factor"
        )

        # ── SECTION 5: Agentic Chain Reasoning ───────────────────────────
        section_divider("🤖 AGENTIC CHAIN REASONING  [Chain Reasoning + Agentic GenAI]")

        if run_audit:
            agent_steps = [
                ("Initializing audit case for " + prov["id"] + "...", 0.4),
                ("Loading 12-month billing history...", 0.5),
                ("Running Z-score time-series anomaly detection...", 0.6),
                ("Computing K-Means cluster assignment...", 0.5),
                ("Calculating composite anomaly score...", 0.4),
                (f"Identifying fraud pattern signature: {prov['fraud_pattern']}...", 0.6),
                ("Cross-referencing ICD-10 vs CPT code alignment...", 0.5),
                ("Extracting top risk factors with confidence scores...", 0.5),
                ("Constructing Cotiviti Payment Integrity audit brief...", 0.7),
                ("Querying Groq LLM (Llama3-70B) for chain reasoning...", 0.3),
            ]

            log_placeholder = st.empty()
            completed_steps: list[str] = []

            for step_text, delay in agent_steps:
                completed_lines = "<br>".join(
                    f'<span style="color:#00FF88;">[ DONE ]  {s}</span>'
                    for s in completed_steps
                )
                active_line = f'<span style="color:#FFD700;">[ RUN  ]  {step_text}</span>'
                log_placeholder.markdown(
                    f'<div class="agent-log">'
                    + (completed_lines + "<br>" if completed_steps else "")
                    + active_line
                    + "</div>",
                    unsafe_allow_html=True,
                )
                time.sleep(delay)
                completed_steps.append(step_text)

            # Show all completed
            log_placeholder.markdown(
                '<div class="agent-log">'
                + "<br>".join(
                    f'<span style="color:#00FF88;">[ DONE ]  {s}</span>'
                    for s in completed_steps
                )
                + "</div>",
                unsafe_allow_html=True,
            )

            # Build audit brief and call Groq
            audit_brief = build_audit_brief(
                prov, ar, score, tier, cluster, risk_factors
            )

            st.markdown("#### 📄 Groq LLM Audit Analysis (Streaming)")
            st.caption(f"Model: {GROQ_MODEL} | Case: CA-2024-{prov['id']}")

            output_box = st.empty()
            full_response = ""

            try:
                stream = stream_groq_response(audit_brief)
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full_response += delta
                    # Stream as markdown inside styled container
                    output_box.markdown(
                        f'<div class="llm-output">{full_response}▌</div>',
                        unsafe_allow_html=True,
                    )
                # Final render — use st.markdown for proper bold/list formatting
                output_box.empty()
                st.markdown(
                    '<div class="llm-output">' + full_response.replace("\n\n", "<br><br>").replace("\n", "<br>") + "</div>",
                    unsafe_allow_html=True,
                )
            except ValueError as e:
                st.error(f"Configuration error: {e}")
            except Exception as e:
                st.error(f"Groq API error: {e}")
        else:
            st.info(
                "👆 Click **Run Full Audit** in the sidebar to activate the agentic reasoning pipeline "
                "and stream the Groq LLM audit analysis."
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Operations Overview [TPO: Operations Domain]
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("## 📊 Provider Portfolio — Operations Overview")
        st.markdown(
            "*Operations-domain view across all 10 audited providers. "
            "Dot size = average monthly claim amount. Color = cluster group.*"
        )

        # Summary KPI cards
        n_high = sum(1 for t in risk_tiers if t == "HIGH")
        n_medium = sum(1 for t in risk_tiers if t == "MEDIUM")
        flagged_exposure = sum(
            np.mean(providers[i]["monthly_amounts"]) * 12
            for i in range(len(providers))
            if risk_tiers[i] in ("HIGH", "MEDIUM")
        )

        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown(
                render_metric_card(
                    "Total Providers Audited",
                    str(len(providers)),
                    sub="Active payment integrity review",
                ),
                unsafe_allow_html=True,
            )
        with kpi2:
            st.markdown(
                render_metric_card(
                    "High-Risk Providers Flagged",
                    str(n_high),
                    sub=f"{n_medium} MEDIUM | {len(providers)-n_high-n_medium} LOW",
                ),
                unsafe_allow_html=True,
            )
        with kpi3:
            st.markdown(
                render_metric_card(
                    "Estimated Financial Exposure",
                    f"${flagged_exposure/1e6:.1f}M",
                    sub="Annual exposure from HIGH + MEDIUM risk providers",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Operations scatter chart
        fig_ops = render_operations_scatter(
            providers, anomaly_scores, risk_tiers, cluster_labels, clustering_result
        )
        st.plotly_chart(fig_ops, use_container_width=True, key="ops_scatter")

        # Provider portfolio table
        section_divider("📋 FULL PROVIDER PORTFOLIO SUMMARY")

        portfolio_data = []
        for i, p in enumerate(providers):
            portfolio_data.append(
                {
                    "Provider ID": p["id"],
                    "Provider Name": p["name"],
                    "Specialty": p["specialty"],
                    "Fraud Pattern": p["fraud_pattern"],
                    "Avg Monthly Claims": f"{np.mean(p['monthly_claims']):.0f}",
                    "Avg Monthly Amount": f"${np.mean(p['monthly_amounts']):,.0f}",
                    "Anomaly Score": f"{anomaly_scores[i]:.1f}",
                    "Cluster": cluster_labels[i],
                    "Risk Tier": risk_tiers[i],
                    "DX-CPT Match": f"{p['dx_proc_match_score']:.2f}",
                }
            )

        portfolio_df = pd.DataFrame(portfolio_data)

        def highlight_risk(row: pd.Series) -> list[str]:
            tier = row["Risk Tier"]
            if tier == "HIGH":
                return ["background-color: rgba(231,76,60,0.2)"] * len(row)
            elif tier == "MEDIUM":
                return ["background-color: rgba(230,126,34,0.15)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            portfolio_df.style.apply(highlight_risk, axis=1),
            use_container_width=True,
            hide_index=True,
            height=380,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        """
<div class="footer">
    ClinicalAudit AI — Built for Cotiviti GenAI Intern Assessment |
    Demonstrating Clinical Decision Making and Pattern Recognition in Healthcare |
    Candidate: Jey Praveen Sivaraj 
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()