"""
app.py — Streamlit UI for the Spot-the-Fake-Photo detector.

Run with:
    streamlit run app.py
"""

import io
import os
import sys
import tempfile
import time
from pathlib import Path


import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── path setup ──────────────────────────────────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from features import extract, NAMES

# ── page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="FakePhoto Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── global background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 10%, #1a1040 0%, #0D0D1A 55%, #0a0818 100%);
}

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(124,92,252,0.25), rgba(200,100,255,0.15));
    border: 1px solid rgba(124,92,252,0.4);
    border-radius: 50px;
    padding: 0.35rem 1.1rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #b39aff;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 30%, #b39aff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.6rem;
    line-height: 1.15;
}
.hero p {
    color: #8888aa;
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── glass card ── */
.glass-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.75rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.25rem;
}

/* ── upload zone override ── */
[data-testid="stFileUploader"] > div {
    background: linear-gradient(135deg, rgba(124,92,252,0.08), rgba(200,100,255,0.04)) !important;
    border: 2px dashed rgba(124,92,252,0.4) !important;
    border-radius: 16px !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(124,92,252,0.8) !important;
    background: linear-gradient(135deg, rgba(124,92,252,0.14), rgba(200,100,255,0.08)) !important;
}

/* ── verdict banner ── */
.verdict-real {
    background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(16,185,129,0.10));
    border: 1px solid rgba(34,197,94,0.35);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.verdict-screen {
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(220,38,38,0.10));
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.verdict-uncertain {
    background: linear-gradient(135deg, rgba(234,179,8,0.18), rgba(202,138,4,0.10));
    border: 1px solid rgba(234,179,8,0.35);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.verdict-emoji { font-size: 3rem; margin-bottom: 0.4rem; }
.verdict-title { font-size: 1.5rem; font-weight: 700; margin: 0 0 0.25rem; }
.verdict-sub   { font-size: 0.9rem; color: #aaaacc; margin: 0; }

/* ── stat pill ── */
.stat-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 0.75rem; }
.stat-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 50px;
    padding: 0.35rem 1rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: #ccccee;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7C5CFC;
    margin-bottom: 0.6rem;
}

/* ── how-it-works grid ── */
.how-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem; }
.how-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.1rem;
}
.how-icon  { font-size: 1.6rem; margin-bottom: 0.5rem; }
.how-title { font-size: 0.85rem; font-weight: 600; color: #ddd; margin-bottom: 0.3rem; }
.how-desc  { font-size: 0.75rem; color: #888; line-height: 1.5; }

/* ── divider ── */
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── model loader ────────────────────────────────────────────────────
def load_model():
    """Always load fresh from disk — avoids stale-cache misclassification after retraining."""
    model_path = HERE / "model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


# ── feature description map ──────────────────────────────────────────
FEATURE_META = {
    "fft_lo":            ("FFT low-freq energy",      "Frequency", "#7C5CFC"),
    "fft_mid":           ("FFT mid-freq energy",      "Frequency", "#7C5CFC"),
    "fft_hi":            ("FFT high-freq energy",     "Frequency", "#7C5CFC"),
    "fft_peak_ratio":    ("FFT peak sharpness",       "Frequency", "#7C5CFC"),
    "hough_line_density":("Straight line density",    "Edges",     "#06b6d4"),
    "hough_rect_score":  ("Rectangular bezel score",  "Edges",     "#06b6d4"),
    "glare_frac":        ("Glare pixel fraction",     "Glare",     "#f59e0b"),
    "glare_blob_frac":   ("Largest glare blob size",  "Glare",     "#f59e0b"),
    "hue_mean":          ("Mean hue",                 "Colour",    "#10b981"),
    "hue_std":           ("Hue spread",               "Colour",    "#10b981"),
    "sat_mean":          ("Mean saturation",          "Colour",    "#10b981"),
    "sat_std":           ("Saturation spread",        "Colour",    "#10b981"),
    "blue_bias":         ("Blue-channel bias",        "Colour",    "#10b981"),
    "lap_var":           ("Laplacian variance",       "Texture",   "#ec4899"),
    "lbp_entropy":       ("LBP texture entropy",      "Texture",   "#ec4899"),
    "lbp_uniformity":    ("LBP uniformity",           "Texture",   "#ec4899"),
    "dct_ac_mean":       ("DCT AC block energy",      "DCT",       "#f97316"),
    "dct_ac_std":        ("DCT AC energy spread",     "DCT",       "#f97316"),
    "dct_ac_skew":       ("DCT AC energy skew",       "DCT",       "#f97316"),
    "edge_density":      ("Canny edge density",       "Edges",     "#06b6d4"),
}


# ── helpers ──────────────────────────────────────────────────────────
def score_to_verdict(score: float):
    if score < 0.35:
        return "real", "Genuine Photo", "✅", "This image shows the physical fingerprints of a real camera shot — no screen artefacts detected.", "verdict-real"
    elif score > 0.65:
        return "screen", "Screen Recapture", "🖥️", "Strong signals of moiré patterns, bezel edges, or glare indicate this is a photo of a screen.", "verdict-screen"
    else:
        return "uncertain", "Uncertain", "⚠️", "The image sits in the ambiguous zone. Consider recapturing with better lighting or different angle.", "verdict-uncertain"


def make_gauge(score: float):
    label, title, _, _, _ = score_to_verdict(score)
    needle_color = "#22c55e" if label == "real" else "#ef4444" if label == "screen" else "#eab308"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#ffffff", "family": "Inter"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "rgba(255,255,255,0.2)",
                "tickfont": {"color": "rgba(255,255,255,0.4)", "size": 10},
                "nticks": 6,
            },
            "bar": {"color": needle_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  35], "color": "rgba(34,197,94,0.20)"},
                {"range": [35, 65], "color": "rgba(234,179,8,0.20)"},
                {"range": [65, 100], "color": "rgba(239,68,68,0.20)"},
            ],
            "threshold": {
                "line": {"color": needle_color, "width": 3},
                "thickness": 0.85,
                "value": score * 100,
            },
        },
        title={"text": "Screen Score", "font": {"size": 13, "color": "#8888aa", "family": "Inter"}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=20, r=20),
        height=220,
        font={"family": "Inter"},
    )
    return fig


def make_feature_chart(vec: np.ndarray):
    # Normalise to 0-1 for display using min-max across features
    v_min, v_max = vec.min(), vec.max()
    norm = (vec - v_min) / (v_max - v_min + 1e-9)

    labels  = [FEATURE_META[n][0] for n in NAMES]
    groups  = [FEATURE_META[n][1] for n in NAMES]
    colors  = [FEATURE_META[n][2] for n in NAMES]

    fig = go.Figure(go.Bar(
        x=norm,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        hovertemplate="<b>%{y}</b><br>Group: " +
                      "<br>Raw value: " +
                      "<extra></extra>",
        text=[f"{v:.3f}" for v in vec],
        textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.55)", size=9, family="Inter"),
        cliponaxis=False,
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=180, r=80),
        height=540,
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
            tickfont=dict(color="rgba(255,255,255,0.3)", size=9),
            title=dict(text="Normalised Value", font=dict(color="#666", size=10)),
        ),
        yaxis=dict(
            tickfont=dict(color="rgba(255,255,255,0.7)", size=10, family="Inter"),
            categoryorder="total ascending",
        ),
        bargap=0.35,
        font={"family": "Inter"},
    )
    return fig


# ── main app ─────────────────────────────────────────────────────────
def main():
    # ── Hero ──
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🔍 AI-Powered Detection</div>
        <h1>FakePhoto Detector</h1>
        <p>Upload any image — our machine learning model analyses 20 physical signals to instantly tell if it's a genuine photo or a screen recapture.</p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("**model.joblib not found.** Run `python train.py` first to train the model.")
        st.stop()

    # ── Layout: left (upload + result) | right (features + info) ──
    col_left, col_right = st.columns([1, 1.05], gap="large")

    with col_left:
        # ── Upload ──
        st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
            help="Supported formats: JPG, PNG, WebP, BMP",
        )

        if uploaded is not None:
            # Save to temp file for OpenCV
            suffix = Path(uploaded.name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())  # getvalue() never exhausts the buffer unlike read()
                tmp_path = tmp.name

            # Show image preview
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Image Preview</div>', unsafe_allow_html=True)
            pil_img = Image.open(tmp_path)
            st.image(pil_img, width='stretch', caption=f"{uploaded.name}  ·  {pil_img.width}×{pil_img.height}px")

            # ── Run analysis ──
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            with st.spinner("Analysing image..."):
                t0  = time.perf_counter()
                vec = extract(tmp_path)
                score = float(model.predict_proba(vec.reshape(1, -1))[0, 1])
                ms  = (time.perf_counter() - t0) * 1000

            os.unlink(tmp_path)

            label, title, emoji, explanation, css_class = score_to_verdict(score)

            # ── Verdict banner ──
            st.markdown(f"""
            <div class="{css_class}">
                <div class="verdict-emoji">{emoji}</div>
                <div class="verdict-title">{title}</div>
                <p class="verdict-sub">{explanation}</p>
            </div>
            """, unsafe_allow_html=True)

            # ── Stats pills ──
            size_kb = len(uploaded.getvalue()) // 1024
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-pill">⚡ {ms:.1f} ms latency</div>
                <div class="stat-pill">📄 {size_kb} KB</div>
                <div class="stat-pill">📐 {pil_img.width}×{pil_img.height}</div>
                <div class="stat-pill">🎯 {score*100:.1f}% screen</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Gauge ──
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Screen Score Gauge</div>', unsafe_allow_html=True)
            st.plotly_chart(make_gauge(score), width='stretch', config={"displayModeBar": False})

            # Store for right column
            st.session_state["vec"]   = vec
            st.session_state["score"] = score
            st.session_state["name"]  = uploaded.name

        else:
            # ── Placeholder ──
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:3rem 1.5rem; color:#555577;">
                <div style="font-size:3rem; margin-bottom:1rem;">📷</div>
                <div style="font-size:1rem; font-weight:600; color:#7777aa;">No image uploaded yet</div>
                <div style="font-size:0.85rem; margin-top:0.5rem;">Drop a photo above to analyse it</div>
            </div>
            """, unsafe_allow_html=True)

    # Compute model display name once — used both in Model Info card and the footer
    _m = load_model()
    _clf_name = type(_m.named_steps["clf"]).__name__ if _m else "Unknown"
    clf_display = {
        "RandomForestClassifier": "Random Forest",
        "LogisticRegression": "Logistic Regression",
        "GradientBoostingClassifier": "Gradient Boosting",
    }.get(_clf_name, _clf_name)
    model_kb = (HERE / "model.joblib").stat().st_size // 1024

    with col_right:
        # ── Feature breakdown ──
        if "vec" in st.session_state:
            vec   = st.session_state["vec"]
            score = st.session_state["score"]

            st.markdown('<div class="section-label">Feature Breakdown</div>', unsafe_allow_html=True)
            st.plotly_chart(make_feature_chart(vec), width='stretch', config={"displayModeBar": False})

            # ── Raw values expander ──
            with st.expander("Raw feature values", expanded=False):
                col_a, col_b = st.columns(2)
                for i, (name, val) in enumerate(zip(NAMES, vec)):
                    meta = FEATURE_META[name]
                    target = col_a if i % 2 == 0 else col_b
                    target.metric(label=meta[0], value=f"{val:.4f}", help=f"Group: {meta[1]}")

        else:
            # ── How it works ──
            st.markdown('<div class="section-label">How It Works</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="how-grid">
                <div class="how-card">
                    <div class="how-icon">🌀</div>
                    <div class="how-title">FFT Frequency Analysis</div>
                    <div class="how-desc">Camera grid + screen pixel grid create moiré → anomalous energy peaks in mid-frequency FFT rings.</div>
                </div>
                <div class="how-card">
                    <div class="how-icon">📐</div>
                    <div class="how-title">Edge & Bezel Detection</div>
                    <div class="how-desc">Screens have hard rectangular bezels — detected via Hough line transforms for long straight edges.</div>
                </div>
                <div class="how-card">
                    <div class="how-icon">💡</div>
                    <div class="how-title">Glare Blob Analysis</div>
                    <div class="how-desc">Glass screens reflect room lights as large near-white blobs. Real objects rarely produce this signature.</div>
                </div>
                <div class="how-card">
                    <div class="how-icon">🎨</div>
                    <div class="how-title">Colour Statistics</div>
                    <div class="how-desc">Backlit RGB subpixels shift hue and bias the blue channel relative to natural-light photography.</div>
                </div>
                <div class="how-card">
                    <div class="how-icon">🧱</div>
                    <div class="how-title">LBP Texture Entropy</div>
                    <div class="how-desc">Organic surfaces produce richer Local Binary Pattern histograms than flat screen fills and gradients.</div>
                </div>
                <div class="how-card">
                    <div class="how-icon">📦</div>
                    <div class="how-title">DCT Double-Compression</div>
                    <div class="how-desc">Re-photographing an already-JPEG-compressed screen leaves a double-banding signature in 8×8 DCT blocks.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Model info card ── (clf_display / model_kb computed above)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Model Info</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="glass-card">
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;">
                    <div>
                        <div style="font-size:0.75rem; color:#666; margin-bottom:0.2rem;">Algorithm</div>
                        <div style="font-weight:600; color:#ddd;">{clf_display}</div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem; color:#666; margin-bottom:0.2rem;">Features</div>
                        <div style="font-weight:600; color:#ddd;">20 hand-crafted</div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem; color:#666; margin-bottom:0.2rem;">Validation</div>
                        <div style="font-weight:600; color:#ddd;">5-fold stratified CV</div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem; color:#666; margin-bottom:0.2rem;">Latency</div>
                        <div style="font-weight:600; color:#ddd;">~20-60 ms / image</div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem; color:#666; margin-bottom:0.2rem;">Model size</div>
                        <div style="font-weight:600; color:#ddd;">{model_kb} KB</div>
                    </div>
                    <div>
                        <div style="font-size:0.75rem; color:#666; margin-bottom:0.2rem;">GPU required</div>
                        <div style="font-weight:600; color:#22c55e;">None &#10003;</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown(f"""
    <hr class="divider">
    <div style="text-align:center; color:#444466; font-size:0.8rem; padding-bottom:1rem;">
        FakePhoto Detector &nbsp;&middot;&nbsp; {clf_display} + 20 hand-crafted CV features &nbsp;&middot;&nbsp; No GPU required
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
