"""
Digital Conservation Suite — Streamlit Dashboard
=================================================
Enter any Met Museum Object ID to run the 4-algorithm pipeline and view
a full conservation report.
"""

import io
import time
import requests
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from algorithms import (
    detect_composition,
    analyze_texture,
    extract_palette,
    classify_style,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Conservation Suite",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark museum aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Mono:wght@300;400&display=swap');

    html, body, [class*="css"] {
        font-family: 'Cormorant Garamond', serif;
        background-color: #fcfaf5;
        color: #2c2c2c;
    }
    .block-container { padding-top: 2rem; max-width: 1400px; }

    h1 { font-size: 2.8rem; font-weight: 300; letter-spacing: 0.12em; color: #d4b896; }
    h2 { font-size: 1.6rem; font-weight: 300; color: #c8b090; letter-spacing: 0.06em; border-bottom: 1px solid #333; padding-bottom: 0.4rem; }
    h3 { font-size: 1.2rem; font-weight: 400; color: #bfa880; }

    /* Optimized Metric Cards for a Light, Modern Gallery Aesthetic */
    .metric-card {
        background: #fcfaf5;
        /* Soft border instead of harsh black */
        border: 1px solid #eceae4; 
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        /* Adds a "floating" effect */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        /* Subtle lift effect on hover */
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.06);
        border-color: #d4af37; /* Gold accent on hover */
    }

    .metric-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        /* Muted charcoal for readability */
        color: #9a9a9a; 
        margin-bottom: 0.5rem;
        display: block;
    }

    .metric-value {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.2rem;
        font-weight: 600;
        /* Deep slate for high contrast */
        color: #1a1a1a; 
        line-height: 1.1;
    }

    .metric-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        /* Subtle olive/gray for secondary info */
        color: #7c7c7c; 
        font-style: italic;
        margin-top: 0.4rem;
    }

    .era-badge {
        display: inline-block;
        background: #fdf8eb; /* Light cream background */
        border: 1px solid #d4b896; /* Soft gold border */
        border-radius: 4px;
        padding: 0.4rem 1rem;
        font-size: 1rem;
        letter-spacing: 0.1em;
        color: #8a6d3b; /* Darker bronze text for visibility */
        font-weight: 600;
    }
    .swatch-strip { border-radius: 3px; margin: 0.5rem 0; }

    .report-section {
        background: #ffffff; /* Pure white for contrast */
        border: 1px solid #eceae4;
        border-left: 4px solid #d4b896; /* Accent gold bar */
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        color: #333333; /* Dark charcoal text */
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .tag {
        display: inline-block;
        background: #1e1e1e;
        border: 1px solid #333;
        padding: 0.1rem 0.6rem;
        border-radius: 2px;
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        color: #aaa;
        margin: 0.2rem;
    }

    stButton>button {
        background: #2a1f0e;
        border: 1px solid #6b4e28;
        color: #e2c98a;
        font-family: 'Cormorant Garamond', serif;
        letter-spacing: 0.08em;
        padding: 0.5rem 2rem;
        font-size: 1rem;
    }
    .sidebar .sidebar-content { background: #fcfaf5; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
MET_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

@st.cache_data(show_spinner=False)
def fetch_object_metadata(object_id: int) -> dict:
    r = requests.get(f"{MET_BASE}/objects/{object_id}", timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def stream_image(url: str) -> np.ndarray:
    r = requests.get(url, timeout=30, stream=True)
    r.raise_for_status()
    arr = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # Resize to max 1200px wide for performance
    h, w = img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        img = cv2.resize(img, (1200, int(h * scale)))
    return img


def hex_swatch_html(palette: list) -> str:
    """Render a coloured swatch strip in HTML."""
    strips = ""
    for p in palette:
        pct = p["percentage"]
        strips += (
            f'<div style="display:inline-block;width:{pct}%;height:48px;'
            f'background:{p["hex"]};title:\'{p["pigment_id"]}\';"></div>'
        )
    return f'<div style="width:100%;border-radius:4px;overflow:hidden;">{strips}</div>'


def pill(text: str, colour: str = "#6b4e28") -> str:
    return (
        f'<span style="background:{colour}22;border:1px solid {colour};'
        f'padding:0.15rem 0.7rem;border-radius:20px;font-size:0.8rem;'
        f'font-family:\'DM Mono\',monospace;color:{colour};margin:0.2rem;">'
        f"{text}</span>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ Conservation Suite")
    st.markdown(
        "<p style='color:#888;font-size:0.9rem;'>"
        "Enter a Met Museum Object ID to run the 4-algorithm pipeline."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    object_id = st.number_input(
        "Met Object ID",
        min_value=1,
        value=436535,
        step=1,
        help="Try 436535 (Van Gogh), 459055 (Monet), 11417 (Rembrandt)",
    )
    run_btn = st.button("▶  Analyse Artwork", use_container_width=True)
    st.markdown("---")
    st.markdown("**Sample IDs to try**")
    examples = {
        "Van Gogh — Wheat Field": 436535,
        "Monet — La Grenouillère": 459055,
        "Vermeer — Young Woman": 670906,
        "Goya — Don Manuel": 10927,
    }
    for label, oid in examples.items():
        if st.button(label, key=f"ex_{oid}"):
            object_id = oid
            run_btn   = True

    st.markdown("---")
    st.markdown(
        "<p style='color:#555;font-size:0.75rem;'>"
        "Data from The Metropolitan Museum of Art Open Access Collection."
        "</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1>🎨 Digital Conservation Suite</h1>"
    "<p style='color:#888;font-size:1.1rem;letter-spacing:0.04em;'>"
    "Multi-algorithm pipeline for art authentication, brushstroke analysis, "
    "pigment extraction, and style classification."
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Fetching artwork from The Met API…"):
        try:
            meta = fetch_object_metadata(int(object_id))
        except Exception as e:
            st.error(f"Could not fetch object {object_id}: {e}")
            st.stop()

    img_url = meta.get("primaryImage") or meta.get("primaryImageSmall")
    if not img_url:
        st.warning(
            "No public image available for this object ID. "
            "Try 436535, 459055, or 670906."
        )
        st.stop()

    # ── Artwork Info Banner ────────────────────────────────────────────────
    col_info, col_img = st.columns([2, 1])
    with col_info:
        st.markdown(f"## {meta.get('title','Untitled')}")
        artist = meta.get("artistDisplayName") or "Unknown Artist"
        date   = meta.get("objectDate") or "Date unknown"
        medium = meta.get("medium") or "Medium unknown"
        dept   = meta.get("department") or ""
        st.markdown(
            f"**Artist:** {artist} &nbsp;·&nbsp; **Date:** {date}  \n"
            f"**Medium:** {medium}  \n"
            f"**Department:** {dept}  \n"
            f"**Object ID:** `{object_id}`",
        )
        if meta.get("objectURL"):
            st.markdown(f"[View on Met Museum →]({meta['objectURL']})")

    with col_img:
        st.image(img_url, use_column_width=True)

    # ── Load image into OpenCV ─────────────────────────────────────────────
    with st.spinner("Streaming high-res image…"):
        try:
            img_bgr = stream_image(img_url)
        except Exception as e:
            st.error(f"Failed to load image: {e}")
            st.stop()

    h_px, w_px = img_bgr.shape[:2]
    st.caption(f"Image resolution: {w_px} × {h_px} px")

    # ── Run the 4 algorithms ───────────────────────────────────────────────
    progress = st.progress(0, text="Initialising pipeline…")
    t0 = time.time()

    progress.progress(5, text="Algorithm 1 / 4 — YOLOv8 Composition Detection…")
    composition = detect_composition(img_bgr)
    progress.progress(30, text="Algorithm 2 / 4 — GLCM Texture Analysis…")
    texture = analyze_texture(img_bgr)
    progress.progress(55, text="Algorithm 3 / 4 — K-Means Palette Extraction…")
    palette_data = extract_palette(img_bgr)
    progress.progress(75, text="Algorithm 4 / 4 — ResNet Style Classification…")
    style = classify_style(img_bgr)
    progress.progress(100, text=f"✓ Pipeline complete in {time.time()-t0:.1f}s")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS DASHBOARD
    # ══════════════════════════════════════════════════════════════════════

    # Row 1 — KPI strip
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Detected Objects</div>"
            f"<div class='metric-value'>{composition['detection_count']}</div>"
            f"<div class='metric-sub'>{composition['composition_type']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Texture Entropy</div>"
            f"<div class='metric-value'>{texture.get('entropy','–')}</div>"
            f"<div class='metric-sub'>{texture.get('inferred_style','–')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with k3:
        dom_hex = palette_data.get("dominant_hex","#888")
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Dominant Pigment</div>"
            f"<div class='metric-value' style='color:{dom_hex};'>{dom_hex}</div>"
            f"<div class='metric-sub'>{palette_data['palette'][0]['pigment_id'] if palette_data.get('palette') else '–'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Artistic Era</div>"
            f"<div class='metric-value' style='font-size:1.1rem;'>{style.get('era_label','–')}</div>"
            f"<div class='metric-sub'>confidence {style.get('confidence',0):.0%}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Algorithm 1 ───────────────────────────────────────────────────────
    st.markdown("## Algorithm 1 — Compositional Layout (YOLOv8)")
    ca, cb = st.columns([3, 2])
    with ca:
        ann = composition.get("annotated_img", img_bgr)
        ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
        st.image(ann_rgb, caption="YOLOv8 detections + Rule-of-Thirds grid",
                 use_column_width=True)
    with cb:
        st.markdown(
            f"<div class='report-section'>"
            f"<div class='metric-label'>Composition Type</div>"
            f"<p style='font-size:1.1rem;'>{composition['composition_type']}</p>"
            f"<div class='metric-label'>Rule-of-Thirds Score</div>"
            f"<p style='font-size:1.1rem;'>{composition['rule_of_thirds_score']} / 1.0</p>"
            f"<div class='metric-label'>Objects Detected</div>"
            f"<p>{composition['detection_count']}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if composition["detections"]:
            st.markdown("**Detected elements:**")
            for det in composition["detections"][:8]:
                st.markdown(
                    f"- `{det['label']}` — conf {det['confidence']:.2f} "
                    f"at ({det['center'][0]}, {det['center'][1]})"
                )

    # ── Algorithm 2 ───────────────────────────────────────────────────────
    st.markdown("## Algorithm 2 — Brushstroke Texture (GLCM)")
    ta, tb = st.columns([2, 3])
    with ta:
        props = texture.get("glcm_properties", {})
        if props:
            for name, vals in props.items():
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>{name.capitalize()}</div>"
                    f"<div class='metric-value' style='font-size:1.3rem;'>{vals['mean']}</div>"
                    f"<div class='metric-sub'>σ = {vals['std']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Shannon Entropy</div>"
            f"<div class='metric-value'>{texture.get('entropy','–')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with tb:
        st.markdown(
            f"<div class='report-section'>"
            f"<div class='metric-label'>Inferred Style</div>"
            f"<p class='era-badge'>{texture.get('inferred_style','–')}</p><br/><br/>"
            f"<p style='font-size:0.95rem;'>{texture.get('style_description','')}</p>"
            f"<hr style='border-color:#333;'/>"
            f"<div class='metric-label'>Dominant Stroke Direction</div>"
            f"<p>{texture.get('dominant_stroke_direction','–')}</p>"
            f"<div class='metric-label'>Roughness Index</div>"
            f"<p>{texture.get('roughness_index','–')}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Algorithm 3 ───────────────────────────────────────────────────────
    st.markdown("## Algorithm 3 — Pigment & Palette Extraction (K-Means)")
    pal = palette_data.get("palette", [])
    if pal:
        st.markdown(hex_swatch_html(pal), unsafe_allow_html=True)
        pa, pb = st.columns([3, 2])
        with pa:
            for p in pal:
                col_dot = f"<span style='color:{p['hex']};font-size:1.4rem;'>■</span>"
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div style='display:flex;align-items:center;gap:0.8rem;'>"
                    f"{col_dot}"
                    f"<div>"
                    f"<div class='metric-label'>Rank {p['rank']} — {p['percentage']}%</div>"
                    f"<div style='font-size:1rem;'><code>{p['hex']}</code> "
                    f"· RGB({p['rgb'][0]}, {p['rgb'][1]}, {p['rgb'][2]})</div>"
                    f"<div style='font-size:0.85rem;color:#888;font-style:italic;'>{p['pigment_id']}</div>"
                    f"</div></div></div>",
                    unsafe_allow_html=True,
                )
        with pb:
            st.markdown(
                f"<div class='report-section'>"
                f"<div class='metric-label'>Palette Warmth</div>"
                f"<p>{palette_data.get('palette_warmth','–')}</p>"
                f"<div class='metric-label'>Average Luminance</div>"
                f"<p>{palette_data.get('average_luminance','–')} / 255</p>"
                f"<div class='metric-label'>Historical Pigments Detected</div>",
                unsafe_allow_html=True,
            )
            for pig in palette_data.get("historical_pigments", []):
                st.markdown(f"- {pig}")
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Algorithm 4 ───────────────────────────────────────────────────────
    st.markdown("## Algorithm 4 — Style & Era Classification (ResNet-50)")
    sa, sb = st.columns([2, 3])
    with sa:
        st.markdown(
            f"<div class='metric-card' style='text-align:center;padding:2rem;'>"
            f"<div class='metric-label'>Classified Era</div>"
            f"<div style='font-size:2rem;font-weight:300;color:#d4b896;margin:0.5rem 0;'>"
            f"{style.get('era_label','–')}</div>"
            f"<div class='metric-label'>Feature Confidence</div>"
            f"<div style='font-size:1.4rem;color:#c8b090;'>{style.get('confidence',0):.1%}</div>"
            f"<div class='metric-sub'>{style.get('device_used','')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with sb:
        st.markdown(
            f"<div class='report-section'>"
            f"<div class='metric-label'>Era Description</div>"
            f"<p>{style.get('era_description','')}</p>"
            f"<hr style='border-color:#333;'/>"
            f"<div class='metric-label'>Top-5 ResNet Feature Activations</div>",
            unsafe_allow_html=True,
        )
        for feat in style.get("top5_imagenet_features", []):
            bar_w = int(feat["probability"] * 100)
            st.markdown(
                f"<div style='margin:0.3rem 0;'>"
                f"<span style='font-family:DM Mono,monospace;font-size:0.75rem;color:#aaa;'>"
                f"{feat['imagenet_class']}</span> — {feat['probability']:.3f}"
                f"<div style='background:#1e1e1e;border-radius:2px;height:4px;margin-top:3px;'>"
                f"<div style='background:#6b4e28;width:{bar_w}%;height:4px;border-radius:2px;'></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Conservation Report Summary ────────────────────────────────────────
    st.divider()
    st.markdown("## 📄 Conservation Report")
    st.markdown(
        f"<div class='report-section'>"
        f"<h3>Executive Summary</h3>"
        f"<p>Object <strong>{object_id}</strong> — "
        f"<em>{meta.get('title','Untitled')}</em> by {meta.get('artistDisplayName','Unknown')}</p>"
        f"<ul>"
        f"<li><strong>Composition:</strong> {composition['composition_type']} "
        f"(rule-of-thirds score: {composition['rule_of_thirds_score']}).</li>"
        f"<li><strong>Brushstroke Texture:</strong> Entropy {texture.get('entropy','N/A')} — "
        f"consistent with <em>{texture.get('inferred_style','unknown')}</em> technique. "
        f"Dominant stroke direction: {texture.get('dominant_stroke_direction','N/A')}.</li>"
        f"<li><strong>Pigment Profile:</strong> Dominant colour {palette_data.get('dominant_hex','N/A')} "
        f"({pal[0]['pigment_id'] if pal else 'N/A'}). "
        f"Palette is {palette_data.get('palette_warmth','N/A')} with average luminance "
        f"{palette_data.get('average_luminance','N/A')}/255.</li>"
        f"<li><strong>Style Classification:</strong> {style.get('era_label','N/A')} — "
        f"{style.get('era_description','')}</li>"
        f"</ul>"
        f"<h3>Conservation Recommendations</h3>"
        f"<p>Based on the above digital analysis, conservators should inspect for:</p>"
        f"<ul>"
        f"<li>Impasto cracking in high-entropy brushstroke zones (entropy > 4.0).</li>"
        f"<li>Pigment degradation in dominant colour regions — "
        f"{pal[0]['pigment_id'] if pal else 'primary pigment'} is historically susceptible to fading.</li>"
        f"<li>Compositional integrity — rule-of-thirds alignment suggests "
        f"{'intentional geometric planning' if composition['rule_of_thirds_score'] > 0.5 else 'organic, less structured layout'}.</li>"
        f"</ul>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        f"Analysis generated by Digital Conservation Suite | "
        f"Pipeline runtime: {time.time()-t0:.1f}s | "
        f"Data: Metropolitan Museum of Art Open Access API"
    )

else:
    st.markdown(
        "<div style='text-align:center;padding:6rem 2rem;color:#444;'>"
        "<div style='font-size:4rem;'>🖼️</div>"
        "<p style='font-size:1.2rem;'>Enter a Met Museum Object ID in the sidebar and click <strong>Analyse Artwork</strong>.</p>"
        "<p>Try Object ID <code>436535</code> — Van Gogh's <em>Wheat Field with Cypresses</em></p>"
        "</div>",
        unsafe_allow_html=True,
    )
