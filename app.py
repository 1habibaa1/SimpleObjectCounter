"""
app.py - Simple Object Counter
================================
A modern Streamlit application for counting objects in images
using digital image processing techniques.

Usage:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import os
import time

from pipeline import (
    preprocess_image,
    segment_image,
    detect_objects,
    count_objects_basic,
    selective_watershed_for_overlapping_objects,
    validate_count,
    extract_features,
    draw_object_ids,
    run_tasks_1_to_4,
)

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Simple Object Counter",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ============================================================
# Helper Functions
# ============================================================

def cv2_to_rgb(img):
    """Convert OpenCV BGR image to RGB for Streamlit display."""
    if len(img.shape) == 2:
        return img  # Grayscale, no conversion needed
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)





def display_image(img, caption=None, use_container_width=True):
    """Display an image with optional caption inside a styled container."""
    rgb = cv2_to_rgb(img)
    st.image(rgb, caption=caption, use_container_width=use_container_width)


def create_area_chart(features_df):
    """Create a styled bar chart for object areas."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=features_df["Object ID"],
        y=features_df["Area (px)"],
        marker=dict(
            color=features_df["Area (px)"],
            colorscale=[[0, "#06B6D4"], [1, "#8B5CF6"]],
            line=dict(width=0),
        ),
        hovertemplate="Object %{x}<br>Area: %{y:.0f} px<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Area per Object", font=dict(size=14, color="#E5E7EB")),
        xaxis=dict(title="Object ID", color="#9CA3AF", gridcolor="rgba(255,255,255,0.05)", dtick=1),
        yaxis=dict(title="Area (px)", color="#9CA3AF", gridcolor="rgba(255,255,255,0.05)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#9CA3AF"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=300,
    )
    return fig


def create_circularity_chart(features_df):
    """Create a styled bar chart for object circularity."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=features_df["Object ID"],
        y=features_df["Circularity"],
        marker=dict(
            color=features_df["Circularity"],
            colorscale=[[0, "#8B5CF6"], [1, "#06B6D4"]],
            line=dict(width=0),
        ),
        hovertemplate="Object %{x}<br>Circularity: %{y:.3f}<extra></extra>",
    ))
    # Perfect circle reference line
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="rgba(245,158,11,0.5)",
        annotation_text="Perfect Circle",
        annotation_font_color="#F59E0B",
        annotation_font_size=11,
    )
    fig.update_layout(
        title=dict(text="Circularity per Object", font=dict(size=14, color="#E5E7EB")),
        xaxis=dict(title="Object ID", color="#9CA3AF", gridcolor="rgba(255,255,255,0.05)", dtick=1),
        yaxis=dict(title="Circularity", color="#9CA3AF", gridcolor="rgba(255,255,255,0.05)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#9CA3AF"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=300,
    )
    return fig


# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="hero-header">
    <div class="hero-icon">🔬</div>
    <h1 class="hero-title">Simple Object Counter</h1>
    <p class="hero-subtitle">Digital Image Processing — Automated Object Detection & Counting</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>🔬 Object Counter</h2>
    </div>
    """, unsafe_allow_html=True)

    # --- Image Upload ---
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">📸 Upload Image</div>', unsafe_allow_html=True)

    img_input = None

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_input = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Validation ---
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">📊 Validation </div>', unsafe_allow_html=True)

    true_count_input = st.number_input(
        "Enter the true object count:",
        min_value=0,
        value=0,
        step=1,
        help="Enter the actual number of objects for accuracy calculation. Leave as 0 to skip.",
    )
    true_count = true_count_input if true_count_input > 0 else None

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Analyze Button ---
    st.markdown("")
    analyze_clicked = st.button("🚀  Analyze Image", use_container_width=True)

    # --- Info ---
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem;">
        <span class="badge badge-cyan">OpenCV</span>&nbsp;
        <span class="badge badge-violet">Watershed</span>&nbsp;
        <span class="badge badge-green">Contours</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Main Content
# ============================================================

# Process image when button is clicked
if analyze_clicked and img_input is not None:
    # Progress animation
    progress_bar = st.progress(0, text="🔬 Initializing pipeline...")

    steps = [
        (15, "📷 Preprocessing image..."),
        (35, "🎯 Segmenting objects..."),
        (55, "🔢 Counting objects..."),
        (75, "📐 Extracting features..."),
        (90, "🎨 Generating visualizations..."),
    ]

    for pct, msg in steps:
        time.sleep(0.3)
        progress_bar.progress(pct, text=msg)

    # Run the full pipeline
    results = run_tasks_1_to_4(
        img=img_input,
        true_count=true_count,
        min_area=100,
    )

    progress_bar.progress(100, text="✅ Analysis complete!")
    time.sleep(0.4)
    progress_bar.empty()

    # Store results in session state
    st.session_state["results"] = results

elif analyze_clicked and img_input is None:
    st.error("⚠️ Please upload an image first.")


# Display results
if "results" in st.session_state:
    results = st.session_state["results"]

    # ---- Metrics Row ----
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card primary">
            <div class="metric-icon">🔢</div>
            <div class="metric-value">{results['final_count']}</div>
            <div class="metric-label">Objects Detected</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">📦</div>
            <div class="metric-value">{results['basic_count']}</div>
            <div class="metric-label">Basic Count</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🔍</div>
            <div class="metric-value">{len(results['suspicious_contours'])}</div>
            <div class="metric-label">Overlaps Found</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Accuracy Card (if validation available) ----
    if results["validation_result"] is not None:
        vr = results["validation_result"]
        acc_color = "#10B981" if vr["Accuracy (%)"] >= 80 else "#F59E0B" if vr["Accuracy (%)"] >= 50 else "#EF4444"
        st.markdown(f"""
        <div class="accuracy-card">
            <div class="accuracy-label">Detection Accuracy</div>
            <div class="accuracy-value" style="color: {acc_color};">{vr['Accuracy (%)']}%</div>
            <div class="accuracy-details">
                <span>True: <strong>{vr['True Count']}</strong></span>
                <span>Predicted: <strong>{vr['Predicted Count']}</strong></span>
                <span>Error: <strong>{vr['Error']}</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ---- Tabs ----
    tab1, tab2, tab3 = st.tabs(["🔬 Pipeline Steps", "🎯 Detection Results", "📐 Object Features"])

    # ======== Tab 1: Pipeline Steps ========
    with tab1:
        pipeline_steps = [
            ("1", "Original Image", "Input image as loaded", results["original"], False),
            ("2", "Grayscale", "Converted to single-channel grayscale", results["gray"], True),
            ("3", "Enhanced", "Brightness & contrast adjusted (α=1.5, β=30)", results["enhanced"], True),
            ("4", "Preprocessed", "Median blur + Gaussian blur applied", results["preprocessed"], True),
            ("5", "Otsu Threshold", "Automatic threshold separating foreground/background", results["otsu_threshold"], True),
            ("6", "Adaptive Threshold", "Local adaptive Gaussian thresholding", results["adaptive_threshold"], True),
            ("7", "Clean Mask", "Morphological opening + closing applied", results["clear_mask"], True),
        ]

        for step_num, title, desc, img, is_gray in pipeline_steps:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-header">
                    <div class="step-number">{step_num}</div>
                    <div>
                        <div class="step-title">{title}</div>
                        <div class="step-desc">{desc}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if is_gray and len(img.shape) == 2:
                st.image(img, use_container_width=True, clamp=True)
            else:
                display_image(img)

    # ======== Tab 2: Detection Results ========
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-header">
                    <div class="step-number">8</div>
                    <div>
                        <div class="step-title">Basic Contour Detection</div>
                        <div class="step-desc">Objects detected using contour analysis — Count: {results['basic_count']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            display_image(results["detected_img"])

        with col2:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-header">
                    <div class="step-number">9</div>
                    <div>
                        <div class="step-title">Hybrid Detection (Watershed)</div>
                        <div class="step-desc">Selective watershed applied for overlaps — Count: {results['final_count']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            display_image(results["hybrid_result"])

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Object IDs
        st.markdown("""
        <div class="step-card">
            <div class="step-header">
                <div class="step-number">★</div>
                <div>
                    <div class="step-title">Final Detection — Objects with IDs</div>
                    <div class="step-desc">Each detected object labeled with a unique identifier</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        display_image(results["object_ids_img"])

    # ======== Tab 3: Object Features ========
    with tab3:
        features_df = results["features_df"]

        if features_df is not None and len(features_df) > 0:
            # Feature table
            st.markdown("""
            <div class="step-card">
                <div class="step-header">
                    <div class="step-number">📋</div>
                    <div>
                        <div class="step-title">Extracted Features Table</div>
                        <div class="step-desc">Area, perimeter, aspect ratio, circularity, and centroid for each object</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(
                features_df,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Charts
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.plotly_chart(
                    create_area_chart(features_df),
                    use_container_width=True,
                )

            with chart_col2:
                st.plotly_chart(
                    create_circularity_chart(features_df),
                    use_container_width=True,
                )
        else:
            st.info("No objects were detected, so no features could be extracted.")


# ---- Welcome State (no results yet) ----
elif "results" not in st.session_state:
    st.markdown("""
    <div class="welcome-card">
        <h3>Welcome! Ready to count some objects? 🎯</h3>
        <p>Upload an image from the sidebar, then click <strong>Analyze Image</strong>.</p>
        <div class="welcome-steps">
            <div class="welcome-step">
                <div class="welcome-step-icon">📸</div>
                <div class="welcome-step-text">Upload Image</div>
            </div>
            <div class="welcome-step">
                <div class="welcome-step-icon">🚀</div>
                <div class="welcome-step-text">Click Analyze</div>
            </div>
            <div class="welcome-step">
                <div class="welcome-step-icon">✨</div>
                <div class="welcome-step-text">View Results</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
