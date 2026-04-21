"""
app.py — Laser Weld Defect Detection Dashboard
LozanoLsa · Project 14 · Co-Training (Semi-Supervised) · 2026

Algorithm: Co-Training · View A: Process params · View B: Sensor monitoring
Domain: Precision Fabrication — Weld Defect Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    recall_score, precision_score, confusion_matrix, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Co-Training · Weld Defect Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── FULL CSS INJECTION ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg:       #080c12;
    --surface:  #0e1420;
    --card:     #121922;
    --card2:    #161f2e;
    --border:   #1e2d45;
    --accent:   #fbbf24;
    --accent2:  #fcd34d;
    --danger:   #f87171;
    --warn:     #fb923c;
    --ok:       #4ade80;
    --blue:     #60a5fa;
    --text:     #c8d8f0;
    --muted:    #4e6a8a;
    --fh: 'Syne', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --fs: 'Instrument Serif', Georgia, serif;
}

.stApp { background: var(--bg) !important; color: var(--text); font-family: var(--fh); }
.block-container { padding: 1.8rem 2.4rem 3rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1rem !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--muted) !important; }
[data-testid="stSidebar"] label { font-family: var(--fm) !important; font-size: 0.7rem !important; color: var(--text) !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }

[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; border: 2px solid var(--accent2) !important; box-shadow: 0 0 8px rgba(251,191,36,0.5) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { font-family: var(--fm) !important; font-size: 0.65rem !important; color: var(--accent2) !important; background: var(--card) !important; border: 1px solid var(--border) !important; padding: 1px 5px !important; border-radius: 3px !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }

[data-testid="stSelectbox"] > div > div { background: var(--card) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: var(--fm) !important; font-size: 0.78rem !important; border-radius: 3px !important; }

[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-top: 2px solid var(--accent) !important; padding: 1rem 1.1rem 0.9rem !important; border-radius: 3px !important; }
[data-testid="stMetricLabel"] > div { font-family: var(--fm) !important; font-size: 0.6rem !important; text-transform: uppercase !important; letter-spacing: 0.18em !important; color: var(--muted) !important; font-weight: 400 !important; }
[data-testid="stMetricValue"] > div { font-family: var(--fm) !important; font-size: 1.7rem !important; font-weight: 600 !important; color: var(--accent2) !important; line-height: 1.1 !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border) !important; gap: 0 !important; background: transparent !important; }
[data-testid="stTabs"] [role="tab"] { font-family: var(--fm) !important; font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; color: var(--muted) !important; padding: 0.5rem 1.2rem !important; border: none !important; border-radius: 0 !important; background: transparent !important; transition: all 0.2s !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--accent2) !important; background: rgba(251,191,36,0.06) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; }
[data-testid="stTabsContent"] { padding-top: 1.4rem !important; }

[data-testid="stAlert"] { border-radius: 2px !important; font-family: var(--fm) !important; font-size: 0.75rem !important; border: none !important; }
[data-testid="stExpander"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stExpander"] summary { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stDataFrame"] th { font-family: var(--fm) !important; font-size: 0.62rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; background: var(--card2) !important; color: var(--muted) !important; }
[data-testid="stDataFrame"] td { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; background: var(--card) !important; }

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--fm) !important; font-size: 0.62rem !important; color: var(--muted) !important; letter-spacing: 0.08em !important; }
h1, h2, h3 { font-family: var(--fh) !important; color: var(--text) !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

.lsa-header { border-bottom: 1px solid var(--border); padding-bottom: 1.2rem; margin-bottom: 0.2rem; }
.lsa-project-tag { font-family: var(--fm); font-size: 0.6rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.22em; margin-bottom: 4px; }
.lsa-title { font-family: var(--fh); font-size: 1.85rem; font-weight: 800; color: #fff; line-height: 1.1; letter-spacing: -0.02em; }
.lsa-tagline { font-family: var(--fs); font-style: italic; font-size: 0.9rem; color: var(--muted); margin-top: 4px; }
.lsa-chip { display: inline-block; background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.3); color: var(--accent2); font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-section { font-family: var(--fm); font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); font-family: var(--fm); font-size: 0.58rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH     = "welding_data.csv"
DATA_PATH_ALT = "14_CoTraining_Weld_Defect/welding_data.csv"
RANDOM_STATE  = 42
FEATURES_A    = ["energy_j", "speed_mm_s", "frequency_khz", "spot_diameter_mm", "gas_flow_l_min"]
FEATURES_B    = ["bead_height_mm", "thermal_var_c", "acoustic_rms", "penetration_mm", "vibration_rms"]
THRESHOLD     = 0.65
ROUNDS        = 5

FEAT_LABELS_A = {
    "energy_j":          "Laser Energy (J)",
    "speed_mm_s":        "Travel Speed (mm/s)",
    "frequency_khz":     "Frequency (kHz)",
    "spot_diameter_mm":  "Spot Diameter (mm)",
    "gas_flow_l_min":    "Gas Flow (L/min)",
}
FEAT_LABELS_B = {
    "bead_height_mm":  "Bead Height (mm)",
    "thermal_var_c":   "Thermal Variance (°C)",
    "acoustic_rms":    "Acoustic RMS",
    "penetration_mm":  "Penetration (mm)",
    "vibration_rms":   "Vibration RMS",
}

# ─── MATPLOTLIB PALETTE ───────────────────────────────────────────────────────
C_BG    = "#080c12"
C_CARD  = "#121922"
C_AMBER = "#fbbf24"
C_AMBER2= "#fcd34d"
C_DANGER= "#f87171"
C_BLUE  = "#60a5fa"
C_OK    = "#4ade80"
C_TEXT  = "#c8d8f0"
C_MUTED = "#4e6a8a"

def dark_fig(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.xaxis.label.set_color(C_MUTED)
    ax.yaxis.label.set_color(C_MUTED)
    ax.title.set_color(C_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2d45")
    return fig, ax

# ─── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for path in [DATA_PATH, DATA_PATH_ALT]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    st.error("welding_data.csv not found. Place the file in the same folder as app.py and restart.")
    st.stop()

@st.cache_resource
def train_models(df):
    df_lab = df[df["is_labeled"] == 1].copy()
    df_unl = df[df["is_labeled"] == 0].copy()

    X_A_lab = df_lab[FEATURES_A]; X_B_lab = df_lab[FEATURES_B]
    y_lab   = df_lab["defect_label"].astype(int)

    X_A_tr, X_A_te, y_tr, y_te = train_test_split(
        X_A_lab, y_lab, test_size=0.2, random_state=RANDOM_STATE, stratify=y_lab
    )
    X_B_tr = df_lab.loc[X_A_tr.index, FEATURES_B]
    X_B_te = df_lab.loc[X_A_te.index, FEATURES_B]
    idx_tr = X_A_tr.index.to_numpy()

    base_A = Pipeline([("sc", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))])
    base_B = Pipeline([("sc", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))])

    cur_A_idx = idx_tr.copy(); cur_A_y = y_tr.values.copy()
    cur_B_idx = idx_tr.copy(); cur_B_y = y_tr.values.copy()
    global_unlab = df_unl.index.to_numpy()

    hist = {"A_f1":[], "B_f1":[], "A_auc":[], "B_auc":[], "A_size":[], "B_size":[]}

    for _ in range(ROUNDS):
        mA = clone(base_A); mB = clone(base_B)
        mA.fit(df.loc[cur_A_idx, FEATURES_A], cur_A_y)
        mB.fit(df.loc[cur_B_idx, FEATURES_B], cur_B_y)

        y_all  = df["defect_real"].values
        pA_all = mA.predict(df[FEATURES_A]); prA_all = mA.predict_proba(df[FEATURES_A])[:, 1]
        pB_all = mB.predict(df[FEATURES_B]); prB_all = mB.predict_proba(df[FEATURES_B])[:, 1]

        hist["A_f1"].append(round(f1_score(y_all, pA_all, zero_division=0), 4))
        hist["B_f1"].append(round(f1_score(y_all, pB_all, zero_division=0), 4))
        hist["A_auc"].append(round(roc_auc_score(y_all, prA_all), 4))
        hist["B_auc"].append(round(roc_auc_score(y_all, prB_all), 4))
        hist["A_size"].append(len(cur_A_idx))
        hist["B_size"].append(len(cur_B_idx))

        pr_A_u = mA.predict_proba(df.loc[global_unlab, FEATURES_A])[:, 1]
        pr_B_u = mB.predict_proba(df.loc[global_unlab, FEATURES_B])[:, 1]

        iA1 = np.where(pr_A_u > THRESHOLD)[0];       iA0 = np.where(pr_A_u < (1 - THRESHOLD))[0]
        iB1 = np.where(pr_B_u > THRESHOLD)[0];       iB0 = np.where(pr_B_u < (1 - THRESHOLD))[0]

        A_ci = np.concatenate([global_unlab[iA1], global_unlab[iA0]])
        A_cy = np.concatenate([np.ones(len(iA1)),  np.zeros(len(iA0))])
        B_ci = np.concatenate([global_unlab[iB1], global_unlab[iB0]])
        B_cy = np.concatenate([np.ones(len(iB1)),  np.zeros(len(iB0))])

        cur_A_idx = np.concatenate([cur_A_idx, B_ci]); cur_A_y = np.concatenate([cur_A_y, B_cy])
        cur_B_idx = np.concatenate([cur_B_idx, A_ci]); cur_B_y = np.concatenate([cur_B_y, A_cy])
        _, ui = np.unique(cur_A_idx, return_index=True); cur_A_idx = cur_A_idx[ui]; cur_A_y = cur_A_y[ui]
        _, ui = np.unique(cur_B_idx, return_index=True); cur_B_idx = cur_B_idx[ui]; cur_B_y = cur_B_y[ui]

    mA_f = clone(base_A); mB_f = clone(base_B)
    mA_f.fit(df.loc[cur_A_idx, FEATURES_A], cur_A_y)
    mB_f.fit(df.loc[cur_B_idx, FEATURES_B], cur_B_y)

    yp_A = mA_f.predict(X_A_te); yp_B = mB_f.predict(X_B_te)
    pr_A = mA_f.predict_proba(X_A_te)[:, 1]; pr_B = mB_f.predict_proba(X_B_te)[:, 1]
    pr_ens = (pr_A + pr_B) / 2; yp_ens = (pr_ens >= 0.50).astype(int)

    def m_dict(yte, yp, proba):
        return {
            "Accuracy" : round(accuracy_score(yte, yp), 4),
            "AUC-ROC"  : round(roc_auc_score(yte, proba), 4),
            "F1"       : round(f1_score(yte, yp, zero_division=0), 4),
            "Recall"   : round(recall_score(yte, yp, zero_division=0), 4),
            "Precision": round(precision_score(yte, yp, zero_division=0), 4),
        }

    metrics = {
        "Model A (Process)": m_dict(y_te, yp_A, pr_A),
        "Model B (Sensors)": m_dict(y_te, yp_B, pr_B),
        "Ensemble":          m_dict(y_te, yp_ens, pr_ens),
    }

    cm_A = confusion_matrix(y_te, yp_A)
    cm_B = confusion_matrix(y_te, yp_B)
    cm_E = confusion_matrix(y_te, yp_ens)
    fpr_A, tpr_A, _ = roc_curve(y_te, pr_A)
    fpr_B, tpr_B, _ = roc_curve(y_te, pr_B)
    fpr_E, tpr_E, _ = roc_curve(y_te, pr_ens)

    return mA_f, mB_f, metrics, cm_A, cm_B, cm_E, fpr_A, tpr_A, fpr_B, tpr_B, fpr_E, tpr_E, hist, y_te

df = load_data()
mA_f, mB_f, metrics, cm_A, cm_B, cm_E, fpr_A, tpr_A, fpr_B, tpr_B, fpr_E, tpr_E, hist, y_te = train_models(df)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="lsa-project-tag">LozanoLsa · Project 14</div>
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
                color:#fff;margin-bottom:6px;">Weld Defect<br>Detector</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                color:#4e6a8a;line-height:1.7;margin-bottom:12px;">
        Co-Training · Logistic Regression<br>Semi-supervised · Paid project
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="lsa-section">// Model info</div>', unsafe_allow_html=True)
    st.caption(f"Algorithm: Co-Training (Semi-Supervised)")
    st.caption(f"View A: Process params → Logistic Regression")
    st.caption(f"View B: Sensor monitoring → Logistic Regression")
    st.caption(f"Labeled welds: {df['is_labeled'].sum():,} (10%)")
    st.caption(f"Unlabeled welds: {(df['is_labeled']==0).sum():,} (90%)")
    st.caption(f"Co-Training rounds: {ROUNDS}")
    st.caption(f"Confidence threshold: {THRESHOLD}")
    st.divider()
    st.caption("Full Project Pack on lozanolsa.gumroad.com")

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #14 · Co-Training · Semi-Supervised · Laser Welding</div>
    <div class="lsa-title">Two Views. One Defect. No Labels Needed.</div>
    <div class="lsa-tagline">Process parameters teach the sensor model. Sensor data teaches the process model. They bootstrap each other across {ROUNDS} rounds.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">CO-TRAINING</span>
        <span class="lsa-chip">VIEW A + VIEW B</span>
        <span class="lsa-chip">ENSEMBLE AUC {metrics['Ensemble']['AUC-ROC']:.4f}</span>
        <span class="lsa-chip">{ROUNDS} ROUNDS · THR {THRESHOLD}</span>
        <span class="lsa-chip">90% UNLABELED</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP KPI ROW ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Ensemble AUC",   f"{metrics['Ensemble']['AUC-ROC']:.4f}",   "Both views combined")
k2.metric("Ensemble Recall",f"{metrics['Ensemble']['Recall']:.4f}",    "Defects caught")
k3.metric("Ensemble F1",    f"{metrics['Ensemble']['F1']:.4f}",        "Precision-recall balance")
k4.metric("Labeled Data",   f"{df['is_labeled'].sum():,} / {len(df):,}","Only 10% inspected")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "PERFORMANCE", "WELD SIMULATOR", "RISK DRIVERS", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    df_lab = df[df["is_labeled"] == 1]
    st.markdown('<div class="lsa-section">// Dataset overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Welds",        f"{len(df):,}")
    c2.metric("Labeled (Inspected)",f"{df['is_labeled'].sum():,}")
    c3.metric("Unlabeled",          f"{(df['is_labeled']==0).sum():,}")
    c4.metric("Global Defect Rate", f"{df['defect_real'].mean()*100:.1f}%")

    st.divider()
    st.markdown('<div class="lsa-section">// Cross-view correlation — View A vs View B</div>',
                unsafe_allow_html=True)
    st.caption("Low correlation = views are independent = Co-Training cross-teaching is meaningful.")
    corr = df[FEATURES_A + FEATURES_B].corr().loc[FEATURES_A, FEATURES_B]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.4, ax=ax, vmin=-1, vmax=1,
                xticklabels=[FEAT_LABELS_B.get(f, f) for f in FEATURES_B],
                yticklabels=[FEAT_LABELS_A.get(f, f) for f in FEATURES_A])
    ax.tick_params(colors=C_MUTED, labelsize=8)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// Thermal variance distribution by quality class — labeled set</div>',
                unsafe_allow_html=True)
    conf_data   = df_lab[df_lab["defect_label"] == 0]["thermal_var_c"]
    defect_data = df_lab[df_lab["defect_label"] == 1]["thermal_var_c"]
    fig2, ax2 = dark_fig(8, 4)
    ax2.hist(conf_data,   bins=20, alpha=0.75, color=C_BLUE,  label="Conforming")
    ax2.hist(defect_data, bins=20, alpha=0.70, color=C_DANGER,label="Defective")
    ax2.axvline(conf_data.mean(),   color=C_BLUE,   linestyle="--", lw=1.8)
    ax2.axvline(defect_data.mean(), color=C_DANGER, linestyle="--", lw=1.8)
    ax2.set_xlabel("Thermal Variance (°C)"); ax2.set_ylabel("Count")
    ax2.legend(labelcolor=C_TEXT, facecolor=C_CARD, fontsize=8)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True); plt.close()
    st.caption("Defective welds show systematically higher thermal variance — the clearest univariate separator in View B.")

    with st.expander("Raw data sample (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// Co-Training model performance — held-out test set</div>',
                unsafe_allow_html=True)
    st.caption("Metrics on labeled records held out before Co-Training began. Ensemble averages both view probabilities.")

    metric_captions = ["% correct predictions", "Ranking ability (1=perfect)",
                       "Harmonic mean P+R", "Defects correctly caught",
                       "Of flagged defects, actually defective"]

    for model_name, m in metrics.items():
        accent = C_AMBER if model_name == "Ensemble" else (C_BLUE if "A" in model_name else C_DANGER)
        st.markdown(f"""
        <div style="font-family:var(--fm);font-size:0.6rem;color:{accent};text-transform:uppercase;
                    letter-spacing:.2em;margin:14px 0 8px;padding-bottom:4px;
                    border-bottom:1px solid {accent}33;">{model_name}</div>
        """, unsafe_allow_html=True)
        cols = st.columns(5)
        for col, (k, v), cap in zip(cols, m.items(), metric_captions):
            col.metric(k, f"{v:.4f}", cap)

    st.divider()
    st.markdown('<div class="lsa-section">// Co-Training convergence across rounds</div>',
                unsafe_allow_html=True)
    rounds = list(range(1, ROUNDS + 1))
    fig3, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig3.patch.set_facecolor(C_BG)
    for ax in axes:
        ax.set_facecolor(C_CARD)
        ax.tick_params(colors=C_MUTED, labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2d45")

    axes[0].plot(rounds, hist["A_f1"], "o-", color=C_BLUE,   lw=2, label="Model A (Process)")
    axes[0].plot(rounds, hist["B_f1"], "o-", color=C_DANGER, lw=2, label="Model B (Sensors)")
    axes[0].set_xlabel("Round", color=C_MUTED); axes[0].set_ylabel("F1", color=C_MUTED)
    axes[0].set_title("F1 per Round", color=C_TEXT)
    axes[0].legend(labelcolor=C_TEXT, facecolor=C_CARD, fontsize=8)
    axes[0].set_xticks(rounds)
    axes[0].xaxis.label.set_color(C_MUTED); axes[0].yaxis.label.set_color(C_MUTED)

    axes[1].plot(rounds, hist["A_auc"], "o-", color=C_BLUE,   lw=2, label="Model A")
    axes[1].plot(rounds, hist["B_auc"], "o-", color=C_DANGER, lw=2, label="Model B")
    axes[1].set_xlabel("Round", color=C_MUTED); axes[1].set_ylabel("AUC", color=C_MUTED)
    axes[1].set_title("AUC-ROC per Round", color=C_TEXT)
    axes[1].legend(labelcolor=C_TEXT, facecolor=C_CARD, fontsize=8)
    axes[1].set_xticks(rounds)
    axes[1].xaxis.label.set_color(C_MUTED); axes[1].yaxis.label.set_color(C_MUTED)

    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True); plt.close()
    st.caption("Each round: Model A's confident predictions label data for Model B, and vice versa. F1 and AUC should rise as the training pool expands.")

    st.divider()
    st.markdown('<div class="lsa-section">// ROC curves — Model A vs B vs Ensemble</div>',
                unsafe_allow_html=True)
    fig4, ax4 = dark_fig(6, 5)
    for name, fpr, tpr, col in [
        ("Model A (Process)", fpr_A, tpr_A, C_BLUE),
        ("Model B (Sensors)", fpr_B, tpr_B, C_DANGER),
        ("Ensemble",          fpr_E, tpr_E, C_AMBER),
    ]:
        auc_val = metrics[name]["AUC-ROC"] if name in metrics else ""
        ax4.plot(fpr, tpr, lw=2, color=col, label=f"{name}  AUC={auc_val:.4f}")
    ax4.plot([0, 1], [0, 1], "--", color=C_MUTED, lw=1)
    ax4.set_xlabel("False Positive Rate"); ax4.set_ylabel("True Positive Rate")
    ax4.legend(labelcolor=C_TEXT, facecolor=C_CARD, fontsize=8)
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True); plt.close()

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="lsa-section">// Weld defect simulator — both views</div>',
                unsafe_allow_html=True)
    st.caption("Enter process and sensor parameters. The Co-Training ensemble estimates defect probability.")

    col_A, col_B, col_res = st.columns([1, 1, 1])

    with col_A:
        st.markdown('<div class="lsa-section">// View A — process parameters</div>',
                    unsafe_allow_html=True)
        energy    = st.slider("Laser Energy (J)",       2.5,  5.5, 4.0, 0.05)
        speed     = st.slider("Travel Speed (mm/s)",   12.0, 30.0, 20.0, 0.5)
        frequency = st.slider("Frequency (kHz)",        5.0, 11.0,  8.0, 0.1)
        spot_diam = st.slider("Spot Diameter (mm)",     0.5,  1.2,  0.8, 0.01)
        gas_flow  = st.slider("Gas Flow (L/min)",       6.0, 18.0, 12.0, 0.5)

    with col_B:
        st.markdown('<div class="lsa-section">// View B — sensor response</div>',
                    unsafe_allow_html=True)
        bead_h      = st.slider("Bead Height (mm)",        0.8,  1.8, 1.2, 0.01)
        therm_v     = st.slider("Thermal Variance (°C)",  10.0, 35.0, 20.0, 0.5)
        acoustic    = st.slider("Acoustic RMS",            0.4,  0.9,  0.6, 0.01)
        penetration = st.slider("Penetration (mm)",        1.2,  3.0,  2.0, 0.05)
        vibration   = st.slider("Vibration RMS",           0.1,  0.6,  0.3, 0.01)

    row_A = pd.DataFrame([{"energy_j": energy, "speed_mm_s": speed, "frequency_khz": frequency,
                            "spot_diameter_mm": spot_diam, "gas_flow_l_min": gas_flow}])
    row_B = pd.DataFrame([{"bead_height_mm": bead_h, "thermal_var_c": therm_v,
                            "acoustic_rms": acoustic, "penetration_mm": penetration,
                            "vibration_rms": vibration}])
    p_A   = mA_f.predict_proba(row_A)[0, 1]
    p_B   = mB_f.predict_proba(row_B)[0, 1]
    p_ens = (p_A + p_B) / 2

    if p_ens >= 0.70:
        risk_label = "HIGH — Consensus Defect"; risk_color = C_DANGER; risk_bg = "#2e0f0f"
    elif p_ens >= 0.45:
        risk_label = "MODERATE — Borderline";   risk_color = C_AMBER;  risk_bg = "#2e2a0a"
    else:
        risk_label = "LOW — Conforming Weld";   risk_color = C_OK;     risk_bg = "#0f2e1a"

    with col_res:
        st.markdown(
            f'''<div style="background:var(--card);border:1px solid var(--border);
                        border-radius:4px;padding:1.4rem 1.6rem;">
                <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
                            color:#fff;margin-bottom:0.8rem;">Risk Assessment</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:3rem;
                            font-weight:700;color:{risk_color};line-height:1;">{p_ens:.1%}</div>
                <div style="margin-top:12px;">
                    <span style="background:{risk_bg};color:{risk_color};
                                 font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                 font-weight:600;letter-spacing:.08em;
                                 padding:4px 12px;border-radius:20px;">{risk_label}</span>
                </div>
                <div style="margin-top:16px;font-family:'JetBrains Mono',monospace;
                            font-size:0.68rem;color:var(--muted);line-height:2.1;">
                    Model A (Process) :
                    <strong style="color:{C_BLUE};">{p_A:.3f}</strong><br>
                    Model B (Sensors) :
                    <strong style="color:{C_DANGER};">{p_B:.3f}</strong><br>
                    Ensemble &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:
                    <strong style="color:{risk_color};">{p_ens:.3f}</strong>
                </div>
            </div>''',
            unsafe_allow_html=True
        )

        # Gauge bar
        fig_g, ax_g = plt.subplots(figsize=(4, 0.8))
        fig_g.patch.set_facecolor(C_BG); ax_g.set_facecolor(C_BG)
        ax_g.barh([0], [1],      color="#1e2d45", height=0.55)
        ax_g.barh([0], [p_ens],  color=risk_color, height=0.55, alpha=0.85)
        ax_g.axvline(0.45, color=C_MUTED, lw=0.8, ls=":")
        ax_g.axvline(0.70, color=C_MUTED, lw=0.8, ls=":")
        ax_g.set_xlim(0, 1); ax_g.axis("off")
        fig_g.tight_layout(pad=0)
        st.pyplot(fig_g, use_container_width=True); plt.close()

        if p_A > 0.5 and p_B > 0.5:
            note = "Both views flag defect — high operational confidence."
        elif p_A > 0.5 or p_B > 0.5:
            note = "Views disagree — consider targeted inspection."
        else:
            note = "Both views indicate conforming weld."
        st.caption(note)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="lsa-section">// Feature coefficients by view</div>',
                unsafe_allow_html=True)
    st.caption("Standardised Logistic Regression coefficients. Red = increases defect risk · Teal = protective.")

    coef_A = mA_f.named_steps["clf"].coef_[0]
    coef_B = mB_f.named_steps["clf"].coef_[0]
    df_cA  = pd.DataFrame({"feature": FEATURES_A, "coef": coef_A}).sort_values("coef", key=abs, ascending=False)
    df_cB  = pd.DataFrame({"feature": FEATURES_B, "coef": coef_B}).sort_values("coef", key=abs, ascending=False)

    col_ca, col_cb = st.columns(2)
    for col, df_c, title, feat_labels in [
        (col_ca, df_cA, "Model A — Process View",  FEAT_LABELS_A),
        (col_cb, df_cB, "Model B — Sensor View",   FEAT_LABELS_B),
    ]:
        with col:
            st.markdown(f'<div class="lsa-section">// {title}</div>', unsafe_allow_html=True)
            for _, row in df_c.iterrows():
                bar_col = C_DANGER if row["coef"] > 0 else C_BLUE
                bar_w   = int(abs(row["coef"]) / df_c["coef"].abs().max() * 180)
                feat_nm = feat_labels.get(row["feature"], row["feature"])
                st.markdown(f"""
                <div style='margin:5px 0; padding:8px 12px; background:var(--card);
                            border-radius:2px; border-left:3px solid {bar_col};'>
                    <div style='font-family:var(--fm); font-size:0.72rem;
                                color:var(--text);'>{feat_nm}</div>
                    <div style='display:flex; align-items:center; gap:8px; margin-top:4px;'>
                        <div style='background:#1e2d45; border-radius:2px; height:7px; width:200px;'>
                            <div style='width:{bar_w}px; background:{bar_col};
                                        border-radius:2px; height:7px;'></div>
                        </div>
                        <span style='font-family:var(--fm); font-size:0.7rem;
                                     color:{bar_col}; font-weight:600;'>{row["coef"]:+.3f}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="lsa-section">// Model agreement analysis — full dataset</div>',
                unsafe_allow_html=True)
    st.caption("When both models agree, the prediction carries the highest operational confidence.")

    yp_A_all = mA_f.predict(df[FEATURES_A])
    yp_B_all = mB_f.predict(df[FEATURES_B])
    both_def = int(((yp_A_all == 1) & (yp_B_all == 1)).sum())
    a_only   = int(((yp_A_all == 1) & (yp_B_all == 0)).sum())
    b_only   = int(((yp_A_all == 0) & (yp_B_all == 1)).sum())
    both_ok  = int(((yp_A_all == 0) & (yp_B_all == 0)).sum())

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, col_css in [
        (c1, "Both: Defect",        both_def, C_DANGER),
        (c2, "Only Process flags",  a_only,   C_AMBER),
        (c3, "Only Sensors flag",   b_only,   C_AMBER),
        (c4, "Both: Conforming",    both_ok,  C_OK),
    ]:
        col.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-top:2px solid {col_css};border-radius:3px;
                    padding:1rem;text-align:center;">
            <div style="font-family:var(--fm);font-size:1.6rem;font-weight:700;
                        color:{col_css};">{val:,}</div>
            <div style="font-family:var(--fm);font-size:0.62rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.1em;margin-top:4px;">{lbl}</div>
        </div>""", unsafe_allow_html=True)

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="lsa-section">// Inspection & action plan by risk tier</div>',
                unsafe_allow_html=True)
    actions = [
        (C_DANGER, "CONSENSUS DEFECT — Both models agree P ≥ 0.70",
         "Both View A and View B independently flag the weld above threshold. Route to immediate "
         "cross-section inspection or radiography before the part advances to the next assembly "
         "station. This category has the lowest false-positive rate."),
        (C_AMBER,  "VIEW DISAGREEMENT — One model flags P ≥ 0.65",
         "Process and sensor data point in different directions. This is the most informative "
         "category: it suggests either a process anomaly not yet reflected in the bead, or a "
         "sensor signal without a clear process cause. Schedule for ultrasound inspection within "
         "the current shift."),
        ("#d4c34a", "MODERATE ENSEMBLE — Avg P between 0.45–0.70",
         "Weld sits in the borderline zone. Add to the priority inspection queue for the shift. "
         "Review the thermal variance reading — it carries the strongest signal in View B. "
         "If > 25°C, escalate immediately."),
        (C_OK,     "LOW RISK — Both models P < 0.45",
         "Weld is within the expected conformance range. Continue monitoring. Re-score if any "
         "process parameters drift by more than 1 sigma from the current shift baseline. "
         "These welds can safely be released."),
        (C_BLUE,   "UNLABELED QUEUE — Not yet inspected",
         "Every physically confirmed inspection on an unlabeled weld adds a real label to the "
         "training pool. Prioritise labeling borderline cases — they deliver the highest model "
         "improvement per inspection cost. Target 15–20 new confirmed labels per month."),
    ]
    for color, tier, text in actions:
        st.markdown(f"""
        <div style='margin:8px 0; padding:1.1rem 1.3rem; background:var(--card);
                    border-radius:2px; border-left:3px solid {color};'>
            <div style='font-family:var(--fm); font-size:0.72rem; font-weight:600;
                        color:{color}; margin-bottom:6px;'>{tier}</div>
            <div style='font-family:var(--fm); font-size:0.7rem; color:var(--muted);
                        line-height:1.7;'>{text}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);
                border-left:3px solid {C_AMBER};border-radius:2px;padding:1rem 1.3rem;">
        <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Deployment reminder</div>
        <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);line-height:1.7;">
            The Co-Training ensemble provides a risk triage score — not an autonomous accept/reject decision.
            Every flagged weld must be dispositioned by a qualified quality technician.
            The model's role is to direct inspection capacity to where it matters most.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:2px;
                padding:1rem 1.3rem;text-align:center;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Full project pack</div>
        <div style="font-family:var(--fm);font-size:0.68rem;color:var(--muted);line-height:1.7;">
            Complete dataset · notebook · presentation deck · simulator
            available on <span style="color:#fbbf24;">lozanolsa.gumroad.com</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Weld Defect Detector · Project 14 · v2.0
</div>
""", unsafe_allow_html=True)
