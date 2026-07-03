# ClinicalAudit AI - Payment Integrity Intelligence System

## From the Candidate

Coming from a background in medical imaging research, building abnormality detection and segmentation systems across multiple modalities along with agentic AI applications, I think about healthcare AI in terms of signal fidelity and false-positive cost. In healthcare AI, a missed abnormality can lead to clinical harm, while a false positive can add unnecessary workflow burden. Payment integrity has the same tradeoff structure: a missed fraud pattern represents financial leakage, but false positives create provider relationship damage and audit cost. That calibration instinct-building systems that distinguish true signal from noise at the feature level, not the threshold level-is what I brought into this system.

I deliberately avoided generic approaches (rule-based flagging, single-metric thresholds) in favor of multivariate statistical methods that mirror how a real analyst actually thinks: combining a time-series signal, a cross-code alignment score, a behavioral rate metric, and a cluster assignment before forming a conclusion. The Groq LLM step is not the intelligence layer - it's the synthesis layer. The intelligence is in the feature engineering.

---

## What the App Does

**ClinicalAudit AI** is a Streamlit-based payment integrity intelligence system that audits synthetic healthcare provider billing profiles for fraud, waste, and abuse patterns. It demonstrates all seven Topic 2 concepts from Cotiviti's GenAI intern assessment in a two-tab professional dashboard:

- **Tab 1 - Audit Interface**: Deep-dive analysis of an individual provider across five labeled sections
- **Tab 2 - Operations Overview**: Portfolio-level view of all 10 providers with financial exposure summary

---

## Demonstration Video

<video width="100%" controls>
  <source src="https://raw.githubusercontent.com/Jeypraveen/Cotiviti-Intern-Assessment-JeyPraveenSivaraj/main/Demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## Topic 2 Concepts Implemented

| Concept | Implementation | Where Visible |
|---|---|---|
| **Time-Series Anomaly Detection** | Z-score normalization on 12-month billing history (numpy). Flags months where \|Z\| > 2.0 relative to provider's own distribution - not absolute thresholds. | Tab 1, Section 1 - Plotly line chart with red anomaly markers and Z-score labels |
| **Clustering** | K-Means (k=4) on 7-dimensional feature vectors. Cluster labels derived from centroid analysis of DX-CPT mismatch, Z-score peaks, and duplicate rates. | Tab 1, Section 2 - Cluster assignment metric card |
| **Classification** | Three-tier risk classification (HIGH/MEDIUM/LOW) from composite anomaly score with domain-calibrated thresholds. | Tab 1, Section 2 - Colored risk badge |
| **Prediction** | Top-3 risk factor extraction with confidence percentages computed from raw feature values via sigmoid-like normalization. | Tab 1, Section 3 - Risk factor table |
| **Inference** | Two-sentence analyst inference synthesizing flag combination into fraud-pattern-specific conclusion. | Tab 1, Section 3 - Inference statement box |
| **Agentic GenAI** | Step-logged orchestration agent with timed delays simulating autonomous audit workflow before LLM call. | Tab 1, Section 5 - Agent activity log |
| **Chain Reasoning** | 5-step structured prompt (flag analysis → pattern confirmation → regulatory context → verdict → actions) to llama-3.3-70b-versatile. | Tab 1, Section 5 - Streaming LLM output |

**TPO Domains covered:**
- **Treatment**: Patient-level care gap risk scoring (Section 4)
- **Payment**: Fraud pattern detection across Sections 1–3
- **Operations**: Portfolio scatter chart and financial exposure summary (Tab 2)

---

## Tech Stack

| Library | Role | Why this choice |
|---|---|---|
| **Streamlit** | UI framework | Fastest path from Python analytics to interactive dashboard; native Plotly integration |
| **Groq (Llama3-70B)** | Chain reasoning LLM | Free tier, sub-second streaming latency, 70B model quality sufficient for structured audit output |
| **scikit-learn** | K-Means clustering + StandardScaler | Production-grade implementation; StandardScaler ensures high-magnitude features (claim amounts) don't dominate clustering |
| **numpy** | Z-score time-series detection | Vectorized computation; ddof=1 for sample std to avoid bias on n=12 months |
| **pandas** | Data handling and table rendering | DataFrame styling API enables risk-tier row highlighting in portfolio table |
| **plotly** | Interactive charts | Hover details, scatter click events, annotation layers - not possible with matplotlib |
| **python-dotenv** | API key management | Prevents hardcoding; standard .env pattern for local development |

---

## Installation

### Windows

```bash
git clone <repository-url>
cd <repository-folder>

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m streamlit run app.py
```

### macOS / Linux

```bash
git clone <repository-url>
cd <repository-folder>

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
pip install -r requirements.txt

python3 -m streamlit run app.py
```

---

## Groq API Key (Free)

1. Go to [https://console.groq.com](https://console.groq.com)
2. Create a free account
3. Navigate to **API Keys** → **Create API Key**
4. Copy the key (shown once)

---

## Configuration

```bash
cp .env.example .env
# Edit .env and paste your key:
# GROQ_API_KEY=gsk_...
```

---

## Running the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Synthetic Data - Fraud Patterns and Why Real Codes Matter

The 10 provider profiles represent named fraud patterns that healthcare payment integrity analysts actually investigate. Using real ICD-10 and CPT codes is not cosmetic - it's what makes the diagnosis-procedure mismatch score meaningful. When the system flags CPT 27447 (total knee arthroplasty) against ICD-10 M54.5 (low back pain), that's a real clinical impossibility that a real LCD policy would deny. A generic "diagnosis_code" vs "procedure_code" with made-up values produces an anomaly score that means nothing.

| Provider | Pattern | Key Signal Codes |
|---|---|---|
| PRV-001 | Upcoding | J06.9 (URI) + CPT 99215 (complex visit) |
| PRV-002 | Unbundling | CPT 29881 + 29882 billed separately |
| PRV-003 | Phantom Billing | Z51.89 + CPT 70553 (MRI brain) |
| PRV-004 | Ping-Ponging | Z00.00 (routine exam) + CPT 99243 (consult) |
| PRV-005 | Credential Billing | C50.911 + CPT 96413 (chemotherapy) |
| PRV-006 | Medically Unnecessary | M54.5 (back pain) + CPT 27447 (knee replacement) |
| PRV-007 | Duplicate Billing | CPT 99283 exact-date duplicates |
| PRV-008 | Seasonal Spike | F32.1 + 8x billing spike months 11–12 |
| PRV-009 | Borderline | Minor elevations across I10, E11.9 |
| PRV-010 | Clean Control | Z00.129 pediatric - all metrics within benchmark |

---

## Design Decisions

**Why Z-score instead of fixed threshold?**
A fixed threshold (e.g., "flag if > 400 claims/month") treats a 400-claim family medicine practice differently than a 400-claim psychiatry practice. Z-score normalization is relative to each provider's own distribution, making the detection scale-invariant. A 4x spike is flagged whether the provider normally bills 100 or 400 claims.

**Why K-Means over DBSCAN or hierarchical clustering?**
K=4 was chosen to match the four distinct fraud risk profiles that domain knowledge suggests exist in any provider portfolio: billing anomaly, code mismatch, phantom/credential risk, and clean. DBSCAN requires tuning epsilon for a 10-provider set which is unstable. Hierarchical clustering produces the same groupings here with more computational overhead for no benefit at this scale.

**Why derive cluster labels from centroids instead of hardcoding?**
Hardcoded labels ("Cluster 0 = fraud") would break if the data distribution shifts. Centroid-derived labeling - assigning "Diagnosis-Procedure Mismatch" to the cluster whose centroid has the lowest DX-CPT alignment score - means the label is earned by the data, not assumed.

**Why 5-step chain reasoning in the Groq prompt?**
Single-question prompts ("is this provider fraudulent?") produce single-answer responses. A 5-step format forces the LLM to reason through the evidence incrementally - the same process a human analyst uses - before reaching a verdict. This is the difference between chain-of-thought and retrieval.
