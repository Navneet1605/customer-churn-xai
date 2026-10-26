"""
app.py
======
Interactive Streamlit Dashboard for Explainable Customer Churn Prediction
and Prescriptive Counterfactual Retention Optimization.

Design: Premium Modern Bright Theme with White-to-Peach & Terracotta Gradients,
Hover-Expandable Sidebar Navigation, High-Contrast Typography, and SHAP Interpretability.
Zero Emojis.
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, List

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.prescriptive_engine import (
    compute_derived_features,
    simulate_intervention,
    generate_prescriptive_packages,
    optimize_minimal_actionable_recourse,
    analyze_cohort_prescriptive_matrix,
    classify_retention_quadrant
)

# Page configuration
st.set_page_config(
    page_title="Churn Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS: Modern Bright Light Theme with Terracotta/Peach Palette
# & Hover-Expandable Sidebar
# ---------------------------------------------------------
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Suppress default Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Global Typography & Colors */
html, body, [class*="css"], .stMarkdown, p, div, span, label {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* App Background: Warm pearl with subtle peach gradient */
.stApp {
    background: linear-gradient(180deg, #fefbf8 0%, #fff7f2 50%, #fef1e8 100%) !important;
    color: #261914 !important;
}

/* Main content container */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1380px;
}

/* ---------------------------------------------------------
   Sidebar: Hover Expandable & Minimizable Navigation
/* ---------------------------------------------------------
   Sidebar: Layered Overlay Navigation (Zero Layout Shift)
--------------------------------------------------------- */
/* Suppress Streamlit's default collapse chevron so sidebar is purely hover-driven */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
button[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* Base fixed sidebar overlay: floats ABOVE the page */
[data-testid="stSidebar"] {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    bottom: 0 !important;
    height: 100vh !important;
    width: 74px !important;
    min-width: 74px !important;
    max-width: 74px !important;
    overflow-x: hidden !important;
    white-space: nowrap !important;
    transition: width 0.3s cubic-bezier(0.2, 0, 0, 1), min-width 0.3s cubic-bezier(0.2, 0, 0, 1), max-width 0.3s cubic-bezier(0.2, 0, 0, 1), box-shadow 0.3s ease !important;
    z-index: 999999 !important;
    background: linear-gradient(180deg, #ffffff 0%, #fff9f5 60%, #fff1e8 100%) !important;
    border-right: 1px solid #fed8c5 !important;
    box-shadow: 2px 0 16px rgba(231, 111, 81, 0.08) !important;
}

/* Expanded state on hover - layers seamlessly above the page without pushing */
[data-testid="stSidebar"]:hover,
[data-testid="stSidebar"]:focus-within {
    width: 320px !important;
    min-width: 320px !important;
    max-width: 320px !important;
    box-shadow: 14px 0 48px rgba(231, 111, 81, 0.22) !important;
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    width: 320px !important;
    min-width: 320px !important;
    overflow-x: hidden !important;
    padding-top: 1.25rem !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
    box-sizing: border-box !important;
}

/* Keep the main page in a fixed, non-reactive position: NEVER shifts or reflows on hover */
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"],
.main {
    margin-left: 74px !important;
    width: calc(100vw - 74px) !important;
    max-width: calc(100vw - 74px) !important;
    transition: none !important;
}

/* ---------------------------------------------------------
   Sidebar Minimized vs Hover Visibility Rules
--------------------------------------------------------- */
/* When sidebar is minimized: hide text, show only symbols */
[data-testid="stSidebar"]:not(:hover) .sidebar-brand-text {
    display: none !important;
}

[data-testid="stSidebar"]:not(:hover) [data-testid="stRadio"] [data-testid="stMarkdownContainer"] {
    display: none !important;
}

[data-testid="stSidebar"]:not(:hover) .sidebar-telemetry {
    display: none !important;
}

/* When sidebar is hovered: smoothly reveal text */
[data-testid="stSidebar"]:hover .sidebar-brand-text {
    display: block !important;
    animation: fadeInText 0.22s ease;
}

[data-testid="stSidebar"]:hover [data-testid="stRadio"] [data-testid="stMarkdownContainer"] {
    display: block !important;
    animation: fadeInText 0.22s ease;
    margin-left: 10px !important;
}

[data-testid="stSidebar"]:hover .sidebar-telemetry {
    display: block !important;
    animation: fadeInText 0.25s ease;
}

@keyframes fadeInText {
    from { opacity: 0; transform: translateX(-6px); }
    to { opacity: 1; transform: translateX(0); }
}

/* ---------------------------------------------------------
   Sidebar Radio Navigation: Symbols & Luxury Item Cards
--------------------------------------------------------- */
/* Force rich dark charcoal typography */
[data-testid="stSidebar"] [data-testid="stRadio"],
[data-testid="stSidebar"] [data-testid="stRadio"] div,
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] div[role="radiogroup"] label p,
[data-testid="stSidebar"] div[role="radiogroup"] label span {
    color: #1a120e !important;
    -webkit-text-fill-color: #1a120e !important;
    font-weight: 700 !important;
    font-size: 0.94rem !important;
    opacity: 1 !important;
    visibility: visible !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
    margin-top: 0.5rem !important;
}

/* Hide default native small radio circle */
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

/* Radio Option Card Base */
[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: #ffffff !important;
    border: 1px solid #f6ded2 !important;
    border-radius: 12px !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    box-shadow: 0 1px 4px rgba(231, 111, 81, 0.04) !important;
    min-height: 44px !important;
}

/* When minimized: card shrinks to a neat 46px square icon container */
[data-testid="stSidebar"]:not(:hover) div[role="radiogroup"] label {
    width: 46px !important;
    min-width: 46px !important;
    height: 46px !important;
    padding: 0 !important;
    justify-content: center !important;
    margin-left: 2px !important;
}

/* When hovered: card expands to full width with comfortable padding */
[data-testid="stSidebar"]:hover div[role="radiogroup"] label {
    width: 100% !important;
    padding: 8px 12px !important;
    justify-content: flex-start !important;
}

/* Basic Symbol Badge for each page item */
[data-testid="stSidebar"] div[role="radiogroup"] label::before {
    content: '' !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-width: 36px !important;
    width: 36px !important;
    height: 36px !important;
    border-radius: 9px !important;
    background-color: #fff8f4 !important;
    border: 1px solid #fed8c5 !important;
    flex-shrink: 0 !important;
    transition: all 0.2s ease !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 19px 19px !important;
}

/* Symbol 1: Executive Overview (Bar chart / telemetry) */
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(1)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%23e76f51' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'%3E%3C/line%3E%3Cline x1='12' y1='20' x2='12' y2='4'%3E%3C/line%3E%3Cline x1='6' y1='20' x2='6' y2='14'%3E%3C/line%3E%3C/svg%3E") !important;
}

/* Symbol 2: Customer Intelligence & XAI (User diagnostic radar) */
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(2)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%23e76f51' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'%3E%3C/path%3E%3Ccircle cx='12' cy='7' r='4'%3E%3C/circle%3E%3C/svg%3E") !important;
}

/* Symbol 3: Prescriptive AI & Retention (Target recourse optimizer) */
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(3)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%230da371' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'%3E%3C/circle%3E%3Ccircle cx='12' cy='12' r='6'%3E%3C/circle%3E%3Ccircle cx='12' cy='12' r='2'%3E%3C/circle%3E%3C/svg%3E") !important;
}

/* Symbol 4: Portfolio Cohort Matrix (2x2 Matrix Grid) */
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(4)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%23f4a261' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='7'%3E%3C/rect%3E%3Crect x='14' y='3' width='7' height='7'%3E%3C/rect%3E%3Crect x='14' y='14' width='7' height='7'%3E%3C/rect%3E%3Crect x='3' y='14' width='7' height='7'%3E%3C/rect%3E%3C/svg%3E") !important;
}

/* Symbol 5: Model Benchmarks & Data (Server Lake Architecture) */
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(5)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%236d5b52' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cellipse cx='12' cy='5' rx='9' ry='3'%3E%3C/ellipse%3E%3Cpath d='M21 12c0 1.66-4 3-9 3s-9-1.34-9-3'%3E%3C/path%3E%3Cpath d='M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5'%3E%3C/path%3E%3C/svg%3E") !important;
}

/* Card hover glow */
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #fff4ed !important;
    border-color: #f4a261 !important;
    transform: translateX(3px) !important;
    box-shadow: 0 4px 14px rgba(231, 111, 81, 0.14) !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover p,
[data-testid="stSidebar"] div[role="radiogroup"] label:hover span {
    color: #e76f51 !important;
    -webkit-text-fill-color: #e76f51 !important;
}

/* Active Selected Radio Item */
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, #fff2eb 0%, #ffe6d9 100%) !important;
    border-color: #e76f51 !important;
    box-shadow: 0 4px 16px rgba(231, 111, 81, 0.2) !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked)::before {
    background-color: #e76f51 !important;
    border-color: #e76f51 !important;
    box-shadow: 0 2px 8px rgba(231, 111, 81, 0.35) !important;
}

/* Selected state white icon overrides */
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(1):has(input:checked)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'%3E%3C/line%3E%3Cline x1='12' y1='20' x2='12' y2='4'%3E%3C/line%3E%3Cline x1='6' y1='20' x2='6' y2='14'%3E%3C/line%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(2):has(input:checked)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2'%3E%3C/path%3E%3Ccircle cx='12' cy='7' r='4'%3E%3C/circle%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(3):has(input:checked)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'%3E%3C/circle%3E%3Ccircle cx='12' cy='12' r='6'%3E%3C/circle%3E%3Ccircle cx='12' cy='12' r='2'%3E%3C/circle%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(4):has(input:checked)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='7'%3E%3C/rect%3E%3Crect x='14' y='3' width='7' height='7'%3E%3C/rect%3E%3Crect x='14' y='14' width='7' height='7'%3E%3C/rect%3E%3Crect x='3' y='14' width='7' height='7'%3E%3C/rect%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(5):has(input:checked)::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cellipse cx='12' cy='5' rx='9' ry='3'%3E%3C/ellipse%3E%3Cpath d='M21 12c0 1.66-4 3-9 3s-9-1.34-9-3'%3E%3C/path%3E%3Cpath d='M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5'%3E%3C/path%3E%3C/svg%3E") !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span {
    color: #b72a00 !important;
    -webkit-text-fill-color: #b72a00 !important;
    font-weight: 800 !important;
}

/* ---------------------------------------------------------
   Widget Styling: High-contrast light inputs & labels
--------------------------------------------------------- */
label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] span,
label[data-testid="stWidgetLabel"],
.stSelectbox label p,
.stSlider label p,
.stMultiSelect label p,
.stTextInput label p,
.stFileUploader label p {
    color: #261914 !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border: 1px solid #eeddd4 !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] * {
    color: #261914 !important;
    background-color: transparent !important;
}

div[data-baseweb="select"]:hover > div,
div[data-baseweb="select"]:focus-within > div {
    border-color: #e76f51 !important;
    box-shadow: 0 0 0 3px rgba(231, 111, 81, 0.16) !important;
}

div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[data-baseweb="menu"] {
    background-color: #ffffff !important;
    border: 1px solid #f6ded2 !important;
    box-shadow: 0 8px 28px rgba(231, 111, 81, 0.14) !important;
    border-radius: 10px !important;
}

li[data-baseweb="menu-item"] {
    color: #261914 !important;
    font-weight: 500 !important;
}

li[data-baseweb="menu-item"]:hover {
    background-color: #fff2eb !important;
    color: #e76f51 !important;
}

/* ---------------------------------------------------------
   Slider Variety: Multi-color Palettes from User Swatches
--------------------------------------------------------- */
.stSlider div[data-baseweb="slider"] div[role="slider"] {
    border: 2px solid #ffffff !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.stSlider div[data-baseweb="slider"] div[role="slider"]:hover {
    transform: scale(1.18) !important;
}

/* Monthly Charges Slider: Golden Apricot / Amber */
.stSlider:has([aria-label*="Monthly Charges"]) div[data-baseweb="slider"] div[role="slider"],
.stSlider:nth-of-type(1) div[data-baseweb="slider"] div[role="slider"] {
    background-color: #f4a261 !important;
    box-shadow: 0 2px 10px rgba(244, 162, 97, 0.5) !important;
}
.stSlider:has([aria-label*="Monthly Charges"]) div[data-baseweb="slider"] > div > div:nth-child(1),
.stSlider:nth-of-type(1) div[data-baseweb="slider"] > div > div:nth-child(1) {
    background: #f4a261 !important;
}
.stSlider:has([aria-label*="Monthly Charges"]) [data-testid="stThumbValue"],
.stSlider:nth-of-type(1) [data-testid="stThumbValue"] {
    color: #d97706 !important;
    font-weight: 700 !important;
}

/* Account Tenure Slider: Eucalyptus / Sage Teal */
.stSlider:has([aria-label*="Tenure"]) div[data-baseweb="slider"] div[role="slider"],
.stSlider:nth-of-type(2) div[data-baseweb="slider"] div[role="slider"] {
    background-color: #2a9d8f !important;
    box-shadow: 0 2px 10px rgba(42, 157, 143, 0.5) !important;
}
.stSlider:has([aria-label*="Tenure"]) div[data-baseweb="slider"] > div > div:nth-child(1),
.stSlider:nth-of-type(2) div[data-baseweb="slider"] > div > div:nth-child(1) {
    background: #2a9d8f !important;
}
.stSlider:has([aria-label*="Tenure"]) [data-testid="stThumbValue"],
.stSlider:nth-of-type(2) [data-testid="stThumbValue"] {
    color: #2a9d8f !important;
    font-weight: 700 !important;
}

/* Target Safety Threshold Slider: Fresh Emerald Mint */
.stSlider:has([aria-label*="Threshold"]) div[data-baseweb="slider"] div[role="slider"] {
    background-color: #0da371 !important;
    box-shadow: 0 2px 10px rgba(13, 163, 113, 0.5) !important;
}
.stSlider:has([aria-label*="Threshold"]) div[data-baseweb="slider"] > div > div:nth-child(1) {
    background: #0da371 !important;
}
.stSlider:has([aria-label*="Threshold"]) [data-testid="stThumbValue"] {
    color: #0da371 !important;
    font-weight: 700 !important;
}

/* Loyalty Discount Slider: Warm Coral Terracotta */
.stSlider:has([aria-label*="Discount"]) div[data-baseweb="slider"] div[role="slider"] {
    background-color: #ff6b4a !important;
    box-shadow: 0 2px 10px rgba(255, 107, 74, 0.5) !important;
}
.stSlider:has([aria-label*="Discount"]) div[data-baseweb="slider"] > div > div:nth-child(1) {
    background: #ff6b4a !important;
}
.stSlider:has([aria-label*="Discount"]) [data-testid="stThumbValue"] {
    color: #d84315 !important;
    font-weight: 700 !important;
}

.stSlider div[data-baseweb="slider"] div[data-testid="stTickBar"] > div {
    background-color: #fed8c5 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #e76f51 0%, #d85c3e 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: 0 4px 14px rgba(231, 111, 81, 0.25) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #f4a261 0%, #e76f51 100%) !important;
    box-shadow: 0 6px 20px rgba(231, 111, 81, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* ---------------------------------------------------------
   Cards & Content Components
--------------------------------------------------------- */
.card-hero-peach {
    background: linear-gradient(135deg, #ffffff 0%, #fff5ee 55%, #ffebe0 100%);
    border: 1px solid #fed8c5;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 6px 24px rgba(231, 111, 81, 0.08);
    margin-bottom: 1.5rem;
}

.metric-card-peach {
    background: #ffffff;
    border: 1px solid #f6ded2;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 10px rgba(231, 111, 81, 0.04);
    transition: all 0.2s ease;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-card-peach:hover {
    transform: translateY(-2px);
    border-color: #e76f51;
    box-shadow: 0 8px 24px rgba(231, 111, 81, 0.16);
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8c786e;
    margin-bottom: 0.3rem;
}
.metric-value-peach {
    font-size: 2rem;
    font-weight: 800;
    color: #1a120e;
    line-height: 1.1;
    letter-spacing: -0.02em;
}

/* Status Badges */
.badge-soft {
    display: inline-block;
    font-size: 0.74rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    margin-top: 0.35rem;
    letter-spacing: 0.02em;
}
.badge-coral {
    background: #fff0eb;
    color: #d84315;
    border: 1px solid #ffd4c4;
}
.badge-emerald {
    background: #ecfdf5;
    color: #2a9d8f;
    border: 1px solid #a7f3d0;
}
.badge-amber {
    background: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
}
.badge-blue {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
}

.insight-callout-peach {
    background: linear-gradient(135deg, #fff7f2 0%, #ffefe5 100%);
    border-left: 4px solid #e76f51;
    border-top: 1px solid #fee3d5;
    border-right: 1px solid #fee3d5;
    border-bottom: 1px solid #fee3d5;
    border-radius: 10px;
    padding: 1.15rem 1.4rem;
    margin: 1.15rem 0;
    color: #3b2c25;
    line-height: 1.6;
    font-size: 0.92rem;
}

.package-card {
    background: #ffffff;
    border: 1px solid #f6ded2;
    border-radius: 14px;
    padding: 1.35rem;
    box-shadow: 0 4px 18px rgba(231, 111, 81, 0.05);
    transition: all 0.2s ease;
    height: 100%;
}
.package-card:hover {
    border-color: #e76f51;
    box-shadow: 0 10px 30px rgba(231, 111, 81, 0.16);
    transform: translateY(-2px);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 1.5rem;
    border-bottom: 2px solid #f6ded2;
}
.stTabs [data-baseweb="tab"] {
    padding-bottom: 0.75rem;
    color: #8c786e;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #d84315 !important;
    border-bottom: 2px solid #e76f51 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #f6ded2;
    border-radius: 12px;
    overflow: hidden;
    background: #ffffff;
}

h1 {
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    color: #1a120e !important;
    margin-bottom: 0.5rem !important;
}
h2 {
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    color: #261914 !important;
    margin-top: 0.5rem !important;
    margin-bottom: 1rem !important;
}
h3 {
    font-weight: 700 !important;
    color: #3b2c25 !important;
    margin-bottom: 0.75rem !important;
}
h4 {
    font-weight: 600 !important;
    color: #4a3b34 !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Functions: Safe HTML Rendering
# ---------------------------------------------------------
def render_html(html_str: str) -> None:
    """Safely renders HTML without markdown parser treating leading indentation as code blocks."""
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


def render_metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    badge_text: str = "",
    badge_type: str = "coral"
) -> str:
    sub_html = f'<div style="font-size: 0.8rem; color: #7c6a60; margin-top: 0.35rem;">{subtitle}</div>' if subtitle else ""
    badge_html = f'<div class="badge-soft badge-{badge_type}">{badge_text}</div>' if badge_text else ""
    return f"""
    <div class="metric-card-peach">
        <div class="metric-label">{title}</div>
        <div class="metric-value-peach">{value}</div>
        {badge_html}
        {sub_html}
    </div>
    """


# ---------------------------------------------------------
# Cached Data Loaders
# ---------------------------------------------------------
@st.cache_data
def load_eda_insights():
    p = BASE_DIR / "reports" / "eda_and_business_insights.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_model_evaluation():
    p = BASE_DIR / "reports" / "model_evaluation_metrics.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_shap_insights():
    p = BASE_DIR / "reports" / "shap_explanations.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_resource
def load_best_model_and_explainer():
    """Loads champion model pipeline and serialized SHAP explainer."""
    model_meta = load_model_evaluation()
    best_file = model_meta.get("best_model_file")
    pipeline = None
    if best_file:
        best_path = Path(best_file)
        if not best_path.is_absolute():
            best_path = BASE_DIR / best_path
        if best_path.exists():
            with open(best_path, "rb") as f:
                pipeline = pickle.load(f)

    explainer_path = BASE_DIR / "models" / "shap_explainer.pkl"
    explainer_obj = None
    if explainer_path.exists():
        with open(explainer_path, "rb") as f:
            explainer_obj = pickle.load(f)

    return pipeline, explainer_obj

@st.cache_data
def load_processed_sample_data():
    p = BASE_DIR / "data" / "processed" / "engineered_telco.parquet"
    if p.exists():
        return pd.read_parquet(p)
    raw_p = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if raw_p.exists():
        return pd.read_csv(raw_p)
    return pd.DataFrame()


# Load artifacts
eda_data = load_eda_insights()
model_eval = load_model_evaluation()
shap_data = load_shap_insights()
champion_pipeline, live_explainer = load_best_model_and_explainer()
df_sample = load_processed_sample_data()


# ---------------------------------------------------------
# Sidebar Navigation & Hover System Telemetry
# ---------------------------------------------------------
with st.sidebar:
    render_html("""
    <div style="display: flex; align-items: center; gap: 12px; padding: 0.25rem 0 1.25rem 0; border-bottom: 1px solid #f6ded2; margin-bottom: 1.25rem;">
        <div style="min-width: 42px; width: 42px; height: 42px; border-radius: 10px; background: linear-gradient(135deg, #e76f51 0%, #d85c3e 100%); display: flex; align-items: center; justify-content: center; color: #ffffff; font-weight: 800; font-size: 1.1rem; box-shadow: 0 4px 12px rgba(231, 111, 81, 0.35); flex-shrink: 0;">
            CI
        </div>
        <div class="sidebar-brand-text" style="overflow: hidden; white-space: nowrap;">
            <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #e76f51; line-height: 1.2;">
                Enterprise Intelligence
            </div>
            <h2 style="font-size: 1.25rem; font-weight: 800; color: #1e1713; margin: 0; letter-spacing: -0.02em; line-height: 1.25;">
                Churn Intelligence
            </h2>
            <div style="display: inline-block; background: #fff2eb; border: 1px solid #fed8c5; border-radius: 5px; padding: 1px 6px; font-size: 0.7rem; font-weight: 700; color: #d84315; margin-top: 0.25rem;">
                PySpark &bull; XAI &bull; Recourse
            </div>
        </div>
    </div>
    """)

    page_selection = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Customer Intelligence & XAI",
            "Prescriptive AI & Retention",
            "Portfolio Cohort Matrix",
            "Model Benchmarks & Data"
        ],
        label_visibility="collapsed"
    )

    render_html("""
    <div class="sidebar-telemetry" style="margin-top: 2rem; padding: 1.15rem; background: #ffffff; border: 1px solid #f6ded2; border-radius: 12px; box-shadow: 0 4px 14px rgba(231, 111, 81, 0.05);">
        <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #8e7a70; margin-bottom: 0.5rem;">
            System Telemetry
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 0.35rem;">
            <span style="color: #6d5b52;">Active Accounts:</span>
            <strong style="color: #1e1713;">7,043</strong>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 0.35rem;">
            <span style="color: #6d5b52;">Storage Architecture:</span>
            <strong style="color: #2a9d8f;">HDFS Parquet</strong>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.82rem;">
            <span style="color: #6d5b52;">Champion ROC-AUC:</span>
            <strong style="color: #e76f51;">0.8464</strong>
        </div>
    </div>
    """)


# =========================================================
# PAGE 1: Executive Overview
# =========================================================
if page_selection == "Executive Overview":
    render_html("""
    <div class="card-hero-peach">
        <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #e76f51; margin-bottom: 0.35rem;">
            Enterprise Telemetry & Strategic Briefing
        </div>
        <h1 style="font-size: 2.3rem; margin: 0; color: #1a120e;">
            Executive Overview & Revenue Exposure
        </h1>
        <p style="color: #6d5a52; font-size: 1rem; margin-top: 0.4rem; max-width: 900px; line-height: 1.5;">
            High-throughput analytics monitoring subscriber health across 7,043 enterprise accounts. 
            Identifies annualized revenue at risk and operational churn catalysts before cancellation occurs.
        </p>
    </div>
    """)

    kpis = eda_data.get("kpis", {})
    best_auc = model_eval.get("best_metrics", {}).get("ROC_AUC", 0.8464)

    # 4 Top Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_html(render_metric_card(
            "Total Subscriber Base",
            f"{kpis.get('total_customers', 7043):,}",
            "Active contracted accounts",
            badge_text="Monitored",
            badge_type="blue"
        ))
    with col2:
        render_html(render_metric_card(
            "Baseline Churn Rate",
            f"{kpis.get('churn_rate_pct', 26.54):.2f}%",
            "Unmanaged annual attrition",
            badge_text="Benchmark",
            badge_type="amber"
        ))
    with col3:
        render_html(render_metric_card(
            "Monthly Revenue at Risk",
            f"${kpis.get('monthly_revenue_at_risk', 139130):,.0f}",
            "30.5% of total gross revenue",
            badge_text="Exposure",
            badge_type="coral"
        ))
    with col4:
        render_html(render_metric_card(
            "Champion Model AUC",
            f"{best_auc:.4f}",
            "Gradient Boosted Trees",
            badge_text="Production",
            badge_type="emerald"
        ))

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Visual Analytics Row
    breakdowns = eda_data.get("breakdowns", {})
    col_v1, col_v2 = st.columns([1, 1.25])

    with col_v1:
        render_html("""
        <div style="margin-bottom: 0.5rem;">
            <h3 style="margin: 0; font-size: 1.15rem;">Revenue & Subscriber Ratio</h3>
            <span style="font-size: 0.82rem; color: #7c6a60;">Gross account attrition distribution</span>
        </div>
        """)
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Retained Accounts', 'At-Risk Churn Accounts'],
            values=[kpis.get('retained_count', 5174), kpis.get('churn_count', 1869)],
            hole=.6,
            marker=dict(colors=['#0da371', '#ff6b4a'], line=dict(color='#ffffff', width=2)),
            textinfo='percent',
            textfont=dict(size=14, color='#ffffff', family='Plus Jakarta Sans')
        )])
        fig_donut.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(color='#261914'))
        )
        st.plotly_chart(fig_donut, width="stretch")

    with col_v2:
        render_html("""
        <div style="margin-bottom: 0.5rem;">
            <h3 style="margin: 0; font-size: 1.15rem;">Attrition by Contractual Commitment</h3>
            <span style="font-size: 0.82rem; color: #7c6a60;">Month-to-month contracts account for 88.5% of churn events</span>
        </div>
        """)
        contract_data = pd.DataFrame(breakdowns.get("by_contract", []))
        if not contract_data.empty:
            fig_contract = px.bar(
                contract_data,
                x="Contract",
                y=["Retained", "Churned"],
                barmode="group",
                color_discrete_map={"Retained": "#0da371", "Churned": "#ff6b4a"},
                height=320
            )
            fig_contract.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, title=None, font=dict(color='#261914')),
                xaxis=dict(gridcolor="#f3e3da", tickfont=dict(color='#261914')),
                yaxis=dict(gridcolor="#f3e3da", tickfont=dict(color='#261914'))
            )
            st.plotly_chart(fig_contract, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    # Secondary Analytics Row
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        render_html("""
        <div style="margin-bottom: 0.5rem;">
            <h3 style="margin: 0; font-size: 1.15rem;">Churn Vulnerability by Internet Service</h3>
            <span style="font-size: 0.82rem; color: #7c6a60;">Fiber optic accounts experience 41.9% churn rate due to missing support</span>
        </div>
        """)
        internet_data = pd.DataFrame(breakdowns.get("by_internet_service", []))
        if not internet_data.empty:
            fig_internet = px.bar(
                internet_data,
                y="InternetService",
                x="ChurnRate",
                orientation='h',
                color_discrete_sequence=['#f4a261'],
                height=260
            )
            fig_internet.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Churn Rate (%)", gridcolor="#f3e3da", tickfont=dict(color='#261914')),
                yaxis=dict(title=None, tickfont=dict(color='#261914'))
            )
            st.plotly_chart(fig_internet, width="stretch")

    with col_b2:
        render_html("""
        <div style="margin-bottom: 0.5rem;">
            <h3 style="margin: 0; font-size: 1.15rem;">Churn Vulnerability by Payment Channel</h3>
            <span style="font-size: 0.82rem; color: #7c6a60;">Electronic check payments exhibit 3x higher churn than automated credit card</span>
        </div>
        """)
        payment_data = pd.DataFrame(breakdowns.get("by_payment_method", []))
        if not payment_data.empty:
            fig_payment = px.bar(
                payment_data,
                y="PaymentMethod",
                x="ChurnRate",
                orientation='h',
                color_discrete_sequence=['#ff6b4a'],
                height=260
            )
            fig_payment.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Churn Rate (%)", gridcolor="#f3e3da", tickfont=dict(color='#261914')),
                yaxis=dict(title=None, tickfont=dict(color='#261914'))
            )
            st.plotly_chart(fig_payment, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    # Architectural Pipeline Banner
    render_html("""
    <div class="insight-callout-peach">
        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #d84315; margin-bottom: 0.35rem;">
            End-to-End Enterprise Big Data Pipeline
        </div>
        <div style="font-weight: 600; font-size: 1rem; color: #261914; margin-bottom: 0.25rem;">
            Distributed Data Lake &bull; PySpark Ingestion &bull; Spark MLlib Pipelines &bull; SHAP Explainability &bull; Prescriptive Recourse Engine
        </div>
        <div style="font-size: 0.85rem; color: #6d5a52;">
            Scalable architecture partitioned by customer commitment in simulated HDFS Parquet tables. 
            Automated model scoring directly triggers minimal-actionable recourse optimization for at-risk accounts.
        </div>
    </div>
    """)


# =========================================================
# PAGE 2: Customer Intelligence & XAI
# =========================================================
elif page_selection == "Customer Intelligence & XAI":
    render_html("""
    <div class="card-hero-peach">
        <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #e76f51; margin-bottom: 0.35rem;">
            Real-Time Scoring & Local Interpretability
        </div>
        <h1 style="font-size: 2.3rem; margin: 0; color: #1a120e;">
            Customer Intelligence & Explainability
        </h1>
        <p style="color: #6d5a52; font-size: 1rem; margin-top: 0.4rem; max-width: 900px; line-height: 1.5;">
            Input or modify individual subscriber parameters to receive immediate churn probability scoring 
            paired with game-theoretic SHAP feature attributions that explain exactly why the customer intends to stay or depart.
        </p>
    </div>
    """)

    if champion_pipeline is None or live_explainer is None:
        st.error("Champion model or SHAP explainer artifacts could not be loaded.")
    else:
        col_input, col_output = st.columns([1, 1.25])

        with col_input:
            render_html("""
            <div style="margin-bottom: 0.75rem;">
                <h3 style="margin: 0; font-size: 1.2rem;">Subscriber Configuration</h3>
                <span style="font-size: 0.82rem; color: #7c6a60;">Adjust account attributes to simulate live inference</span>
            </div>
            """)

            with st.container():
                st.markdown("#### Contractual & Billing Profile")
                f_contract = st.selectbox("Contractual Agreement", ["Month-to-month", "One year", "Two year"], index=0)
                f_payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=0)
                f_monthly = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=89.5, step=0.5)
                f_tenure = st.slider("Account Tenure (Months)", min_value=1, max_value=72, value=4)

                st.markdown("#### Service Subscriptions & Add-ons")
                f_internet = st.selectbox("Internet Service Infrastructure", ["Fiber optic", "DSL", "No"], index=0)
                
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    f_tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0)
                    f_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0)
                    f_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=0)
                with col_s2:
                    f_streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=1)
                    f_streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=1)
                    f_paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)

                f_gender = "Female"
                f_senior = 0
                f_partner = "No"
                f_dependents = "No"
                f_phone = "Yes"
                f_lines = "No"
                f_device_prot = "No"

        raw_customer_dict = {
            "gender": f_gender, "SeniorCitizen": f_senior, "Partner": f_partner, "Dependents": f_dependents,
            "PhoneService": f_phone, "MultipleLines": f_lines, "InternetService": f_internet,
            "OnlineSecurity": f_security, "OnlineBackup": f_backup, "DeviceProtection": f_device_prot,
            "TechSupport": f_tech_support, "StreamingTV": f_streaming_tv, "StreamingMovies": f_streaming_movies,
            "Contract": f_contract, "PaperlessBilling": f_paperless, "PaymentMethod": f_payment,
            "tenure": f_tenure, "MonthlyCharges": f_monthly
        }

        full_feature_dict = compute_derived_features(raw_customer_dict)
        input_df = pd.DataFrame([full_feature_dict])

        with col_output:
            churn_prob = float(champion_pipeline.predict_proba(input_df)[0, 1])

            # Determine dynamic status styling
            if churn_prob >= 0.55:
                risk_tier = "Critical Flight Risk"
                risk_color = "#e76f51"
                badge_type = "coral"
            elif churn_prob >= 0.35:
                risk_tier = "Moderate Risk (Vulnerable)"
                risk_color = "#f4a261"
                badge_type = "amber"
            else:
                risk_tier = "Stable (Low Risk)"
                risk_color = "#0da371"
                badge_type = "emerald"

            render_html(f"""
            <div style="margin-bottom: 0.75rem;">
                <h3 style="margin: 0; font-size: 1.2rem;">Diagnostic Scorecard</h3>
                <span style="font-size: 0.82rem; color: #7c6a60;">Production model confidence & risk classification</span>
            </div>
            """)

            # Speedometer Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                number={'suffix': "%", 'font': {'size': 44, 'color': risk_color, 'family': 'Plus Jakarta Sans', 'weight': 800}},
                title={'text': f"Predicted Churn Propensity<br><span style='font-size:0.85rem;color:#7c6a60;'>Status: {risk_tier}</span>", 'font': {'size': 16, 'color': '#1f1612', 'family': 'Plus Jakarta Sans'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#8c786e', 'tickwidth': 1},
                    'bar': {'color': risk_color, 'thickness': 0.28},
                    'bgcolor': '#fff7f2',
                    'borderwidth': 1,
                    'bordercolor': '#f6ded2',
                    'steps': [
                        {'range': [0, 35], 'color': '#ecfdf5'},
                        {'range': [35, 60], 'color': '#fffbeb'},
                        {'range': [60, 100], 'color': '#fff1f2'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_gauge, width="stretch")

            # Local SHAP Risk Drivers Bar Chart
            st.markdown("#### Primary Churn Drivers (Local SHAP Attribution)")
            preproc = live_explainer["preprocessor"]
            explainer = live_explainer["explainer"]
            readable_names = live_explainer["readable_feature_names"]

            X_single_trans = preproc.transform(input_df)
            raw_shap = explainer.shap_values(X_single_trans)

            if isinstance(raw_shap, list) and len(raw_shap) == 2:
                single_shap = raw_shap[1][0]
            elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
                single_shap = raw_shap[0, :, 1]
            elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
                single_shap = raw_shap[0]
            else:
                single_shap = raw_shap

            paired_shap = list(zip(readable_names, single_shap))
            top_contributors = sorted(paired_shap, key=lambda x: abs(x[1]), reverse=True)[:7]
            top_feats_df = pd.DataFrame(top_contributors, columns=["Feature", "SHAP Impact"])
            # Palette mapping from user swatch: distinct tonal hues across top drivers
            coral_palette = ["#d84315", "#ff6b4a", "#f28e79", "#f4a261", "#e99d42", "#fac984", "#fed6a4"]
            emerald_palette = ["#0da371", "#2a9d8f", "#52b788", "#74c69d", "#84a98c", "#a3b18a", "#b7e4c7"]

            pos_idx = 0
            neg_idx = 0
            bar_colors = []
            for val in top_feats_df["SHAP Impact"]:
                if val >= 0:
                    bar_colors.append(coral_palette[min(pos_idx, len(coral_palette) - 1)])
                    pos_idx += 1
                else:
                    bar_colors.append(emerald_palette[min(neg_idx, len(emerald_palette) - 1)])
                    neg_idx += 1

            top_feats_df["BarColor"] = bar_colors

            fig_shap_bar = go.Figure(go.Bar(
                x=top_feats_df["SHAP Impact"],
                y=top_feats_df["Feature"],
                orientation='h',
                marker=dict(
                    color=top_feats_df["BarColor"],
                    line=dict(color='#ffffff', width=1.5)
                ),
                hovertemplate="<b>%{y}</b><br>Marginal SHAP: %{x:+.3f}<extra></extra>"
            ))
            fig_shap_bar.update_layout(
                yaxis={'categoryorder': 'total ascending', 'tickfont': {'color': '#261914', 'family': 'Plus Jakarta Sans', 'size': 12}},
                xaxis={'title': "Marginal SHAP Contribution (Log-Odds Impact)", 'gridcolor': '#f3e3da', 'tickfont': {'color': '#261914', 'family': 'Plus Jakarta Sans'}},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=290,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_shap_bar, width="stretch")

            # Local SHAP Educational Explanation Box
            render_html("""
            <div style="background: #ffffff; border: 1px solid #f6ded2; border-radius: 14px; padding: 1.25rem 1.4rem; margin-top: 1rem; box-shadow: 0 4px 16px rgba(231, 111, 81, 0.05);">
                <div style="font-size: 0.8rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; color: #e76f51; margin-bottom: 0.6rem;">
                    SHAP Interpretability Guide &mdash; How to Read This Customer's Attribution Breakdown
                </div>
                <div style="font-size: 0.86rem; color: #3b2c25; line-height: 1.65;">
                    <p style="margin: 0 0 0.65rem 0;">
                        <strong>What is SHAP?</strong> SHAP (SHapley Additive exPlanations) is rooted in cooperative game theory. 
                        Instead of treating the model as a black box, it decomposes the prediction into the exact additive contribution of each contract and service feature for this customer relative to the portfolio baseline churn rate (26.5%).
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 0.75rem; margin-top: 0.5rem;">
                        <div style="background: #fff5f2; border: 1px solid #ffd8cc; border-radius: 8px; padding: 0.85rem;">
                            <strong style="color: #ff6b4a; font-size: 0.88rem;">Coral / Orange Bars (Risk Drivers):</strong><br>
                            <span style="font-size: 0.82rem; color: #5a463e;">
                                Features pushing this subscriber's flight risk <strong>higher</strong> (e.g., month-to-month contract, missing technical support, or high monthly invoice). These are the prime targets for prescriptive counterfactual intervention.
                            </span>
                        </div>
                        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 0.85rem;">
                            <strong style="color: #0da371; font-size: 0.88rem;">Emerald / Green Bars (Retention Anchors):</strong><br>
                            <span style="font-size: 0.82rem; color: #2e4a3d;">
                                Features pulling risk <strong>safely downward</strong> away from cancellation (e.g., long account tenure, automated bank transfers, or active cybersecurity suites).
                            </span>
                        </div>
                        <div style="background: #fffbf0; border: 1px solid #fde68a; border-radius: 8px; padding: 0.85rem;">
                            <strong style="color: #b45309; font-size: 0.88rem;">Bar Magnitude (Length):</strong><br>
                            <span style="font-size: 0.82rem; color: #5a463e;">
                                The horizontal span indicates the <strong>exact mathematical strength</strong> of that feature. Longer bars represent the highest-leverage operational catalysts for this account.
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            """)

            if churn_prob >= 0.35:
                render_html(f"""
                <div class="insight-callout-peach" style="margin-top: 1rem;">
                    <strong>Prescriptive Intervention Recommended:</strong> This subscriber demonstrates a {churn_prob*100:.1f}% churn risk driven by contract volatility and service friction. 
                    Navigate to the <strong>Prescriptive AI & Retention</strong> page to solve for the minimal-cost retention policy.
                </div>
                """)


# =========================================================
# PAGE 3: Prescriptive AI & Retention
# =========================================================
elif page_selection == "Prescriptive AI & Retention":
    render_html("""
    <div class="card-hero-peach">
        <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #e76f51; margin-bottom: 0.35rem;">
            Algorithmic Recourse & Prescriptive Intelligence
        </div>
        <h1 style="font-size: 2.3rem; margin: 0; color: #1a120e;">
            Prescriptive Counterfactual Retention Engine
        </h1>
        <p style="color: #6d5a52; font-size: 1rem; margin-top: 0.4rem; max-width: 900px; line-height: 1.5;">
            Moving beyond passive prediction. This engine formulates minimal-cost actionable counterfactual interventions 
            (Wachter framework) to calculate the precise commercial offers that flip high-risk subscribers into loyal accounts with proven ROI.
        </p>
    </div>
    """)

    if champion_pipeline is None:
        st.error("Champion model pipeline not loaded.")
    else:
        # Profile Configuration Row
        col_p1, col_p2, col_p3 = st.columns([1.2, 1, 1])
        with col_p1:
            persona_choice = st.selectbox(
                "Benchmark Target Persona",
                [
                    "High Flight Risk (Month-to-month, Fiber, No Support)",
                    "Support Deficient High Spender ($95/mo, Multiple Lines)",
                    "Price Sensitive Senior ($68/mo, Electronic Check)",
                    "Custom Profile Sandbox"
                ]
            )

        if "High Flight" in persona_choice:
            p_contract = "Month-to-month"
            p_pay = "Electronic check"
            p_internet = "Fiber optic"
            p_tenure = 3
            p_monthly = 89.5
            p_tech = "No"
            p_sec = "No"
            p_backup = "No"
        elif "Support Deficient" in persona_choice:
            p_contract = "Month-to-month"
            p_pay = "Bank transfer (automatic)"
            p_internet = "Fiber optic"
            p_tenure = 8
            p_monthly = 95.0
            p_tech = "No"
            p_sec = "No"
            p_backup = "Yes"
        elif "Price Sensitive" in persona_choice:
            p_contract = "Month-to-month"
            p_pay = "Electronic check"
            p_internet = "DSL"
            p_tenure = 12
            p_monthly = 68.0
            p_tech = "No"
            p_sec = "No"
            p_backup = "No"
        else:
            p_contract = "Month-to-month"
            p_pay = "Electronic check"
            p_internet = "Fiber optic"
            p_tenure = 4
            p_monthly = 85.0
            p_tech = "No"
            p_sec = "No"
            p_backup = "No"

        with col_p2:
            target_threshold = st.slider("Target Safety Threshold (% Max Risk)", min_value=15, max_value=40, value=30, step=5) / 100.0

        with col_p3:
            render_html(f"""
            <div style="background: #ffffff; border: 1px solid #f6ded2; border-radius: 12px; padding: 0.9rem 1.15rem; margin-top: 0.25rem;">
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #8e7a70;">Optimization Constraint</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #d84315;">Reduce Risk &le; {target_threshold * 100:.0f}%</div>
                <div style="font-size: 0.78rem; color: #7c6a60; margin-top: 0.2rem;">Minimizing total promotional cost</div>
            </div>
            """)

        target_customer = {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "PhoneService": "Yes", "MultipleLines": "No", "InternetService": p_internet,
            "OnlineSecurity": p_sec, "OnlineBackup": p_backup, "DeviceProtection": "No",
            "TechSupport": p_tech, "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": p_contract, "PaperlessBilling": "Yes", "PaymentMethod": p_pay,
            "tenure": p_tenure, "MonthlyCharges": p_monthly
        }

        # 1. Run Algorithmic Minimal Actionable Recourse Search
        recourse_result = optimize_minimal_actionable_recourse(
            target_customer, champion_pipeline, target_risk=target_threshold
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Algorithmic Minimal Actionable Recourse (Solver Output)")

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        base_p = recourse_result["baseline_probability"]
        new_p = recourse_result["intervened_probability"]
        net_saved = recourse_result["net_value_saved"]
        roi_rec = recourse_result["roi_pct"]

        with col_r1:
            render_html(render_metric_card(
                "Baseline Churn Risk",
                f"{base_p * 100:.1f}%",
                "Unmanaged risk state",
                badge_text="Initial State",
                badge_type="coral"
            ))
        with col_r2:
            render_html(render_metric_card(
                "Optimized Recourse Risk",
                f"{new_p * 100:.1f}%",
                f"Absolute drop: {(base_p - new_p) * 100:.1f}%",
                badge_text="Target Met",
                badge_type="emerald"
            ))
        with col_r3:
            render_html(render_metric_card(
                "Annual Retention Cost",
                f"${recourse_result['annual_cost']:.2f}",
                "Discounts & service subsidies",
                badge_text="Budget Needed",
                badge_type="blue"
            ))
        with col_r4:
            render_html(render_metric_card(
                "Net Return on Spend (ROI)",
                f"{roi_rec:.1f}%",
                f"Net Saved: ${net_saved:,.2f}/yr",
                badge_text="High Return",
                badge_type="emerald"
            ))

        # Show recommended actionable levers
        rec_interventions = recourse_result.get("recommended_interventions", {})
        if rec_interventions:
            render_html(f"""
            <div class="insight-callout-peach">
                <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #d84315; margin-bottom: 0.35rem;">
                    Optimal Prescriptive Action Policy
                </div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #1e1713; margin-bottom: 0.5rem;">
                    {recourse_result.get('message', '')}
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-top: 0.5rem;">
                    <div style="background: #ffffff; padding: 0.6rem 0.8rem; border-radius: 8px; border: 1px solid #fedecd;">
                        <span style="font-size: 0.75rem; color: #8e7a70;">Contract Duration:</span><br>
                        <strong style="color: #d84315;">{rec_interventions.get('Contract', p_contract)}</strong>
                    </div>
                    <div style="background: #ffffff; padding: 0.6rem 0.8rem; border-radius: 8px; border: 1px solid #fedecd;">
                        <span style="font-size: 0.75rem; color: #8e7a70;">Dedicated Tech Support:</span><br>
                        <strong style="color: #2a9d8f;">{rec_interventions.get('TechSupport', p_tech)}</strong>
                    </div>
                    <div style="background: #ffffff; padding: 0.6rem 0.8rem; border-radius: 8px; border: 1px solid #fedecd;">
                        <span style="font-size: 0.75rem; color: #8e7a70;">Digital Security Suite:</span><br>
                        <strong style="color: #2a9d8f;">{rec_interventions.get('OnlineSecurity', p_sec)}</strong>
                    </div>
                    <div style="background: #ffffff; padding: 0.6rem 0.8rem; border-radius: 8px; border: 1px solid #fedecd;">
                        <span style="font-size: 0.75rem; color: #8e7a70;">Payment Modernization:</span><br>
                        <strong style="color: #1e1713;">{rec_interventions.get('PaymentMethod', p_pay)}</strong>
                    </div>
                    <div style="background: #ffffff; padding: 0.6rem 0.8rem; border-radius: 8px; border: 1px solid #fedecd;">
                        <span style="font-size: 0.75rem; color: #8e7a70;">Monthly Loyalty Concession:</span><br>
                        <strong style="color: #d84315;">${rec_interventions.get('monthly_discount_usd', 0.0):.2f}/mo ({rec_interventions.get('discount_pct', 0.0):.0f}%)</strong>
                    </div>
                </div>
            </div>
            """)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # SECTION 2: Interactive "What-If" Counterfactual Sandbox (MOVED ABOVE PACKAGES)
        # ---------------------------------------------------------
        st.subheader("Interactive 'What-If' Simulation Sandbox")
        with st.container():
            col_sb1, col_sb2 = st.columns([1, 1.25])

            with col_sb1:
                render_html("""
                <div style="margin-bottom: 0.5rem;">
                    <h4 style="margin: 0; font-size: 1.05rem;">Simulate Custom Interventions</h4>
                    <span style="font-size: 0.82rem; color: #7c6a60;">Test hypothetical promotional mixes</span>
                </div>
                """)
                sim_contract = st.selectbox("Proposed Contract Term", ["Month-to-month", "One year", "Two year"], index=1)
                sim_payment = st.selectbox("Proposed Payment Channel", ["Electronic check", "Bank transfer (automatic)", "Credit card (automatic)"], index=1)
                
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    sim_tech = st.selectbox("Bundle Tech Support", ["No", "Yes"], index=1)
                    sim_security = st.selectbox("Bundle Online Security", ["No", "Yes"], index=1)
                with col_t2:
                    sim_discount = st.slider("Monthly Loyalty Discount ($)", min_value=0.0, max_value=25.0, value=10.0, step=2.5)

                custom_sim = simulate_intervention(
                    target_customer,
                    {
                        "Contract": sim_contract,
                        "PaymentMethod": sim_payment,
                        "TechSupport": sim_tech,
                        "OnlineSecurity": sim_security
                    },
                    champion_pipeline,
                    monthly_discount_usd=sim_discount,
                    service_subsidy_cost_monthly=3.0 if (sim_tech == "Yes" or sim_security == "Yes") else 0.0
                )

            with col_sb2:
                render_html("""
                <div style="margin-bottom: 0.5rem;">
                    <h4 style="margin: 0; font-size: 1.05rem;">Real-Time Recourse Evaluation</h4>
                    <span style="font-size: 0.82rem; color: #7c6a60;">Pre vs Post intervention risk & net unit economics</span>
                </div>
                """)
                
                sb_base_p = custom_sim["baseline_probability"]
                sb_new_p = custom_sim["intervened_probability"]

                fig_dual_gauge = go.Figure()
                fig_dual_gauge.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=sb_base_p * 100,
                    number={'suffix': "%", 'font': {'size': 26, 'color': '#ff6b4a', 'family': 'Plus Jakarta Sans', 'weight': 700}},
                    title={'text': "Pre-Intervention", 'font': {'size': 14, 'color': '#7c6a60', 'family': 'Plus Jakarta Sans'}},
                    domain={'x': [0, 0.48], 'y': [0, 1]},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#ff6b4a', 'thickness': 0.3}, 'bgcolor': '#fff5f0'}
                ))
                fig_dual_gauge.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=sb_new_p * 100,
                    number={'suffix': "%", 'font': {'size': 26, 'color': '#0da371', 'family': 'Plus Jakarta Sans', 'weight': 700}},
                    title={'text': "Post-Intervention", 'font': {'size': 14, 'color': '#7c6a60', 'family': 'Plus Jakarta Sans'}},
                    domain={'x': [0.52, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#0da371', 'thickness': 0.3}, 'bgcolor': '#ecfdf5'}
                ))
                fig_dual_gauge.update_layout(
                    height=200,
                    margin=dict(l=10, r=10, t=25, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_dual_gauge, width="stretch")

                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    render_html(f"""
                    <div style="background: #ffffff; border: 1px solid #f6ded2; border-radius: 10px; padding: 0.75rem; text-align: center;">
                        <span style="font-size: 0.72rem; color: #8c786e; text-transform: uppercase; font-weight: 700;">Net Preserved Value</span>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #0da371;">${custom_sim['net_value_saved']:.2f}/yr</div>
                    </div>
                    """)
                with col_res2:
                    render_html(f"""
                    <div style="background: #ffffff; border: 1px solid #f6ded2; border-radius: 10px; padding: 0.75rem; text-align: center;">
                        <span style="font-size: 0.72rem; color: #8c786e; text-transform: uppercase; font-weight: 700;">Intervention Cost</span>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #1e1713;">${custom_sim['annual_intervention_cost']:.2f}/yr</div>
                    </div>
                    """)
                with col_res3:
                    render_html(f"""
                    <div style="background: #ffffff; border: 1px solid #f6ded2; border-radius: 10px; padding: 0.75rem; text-align: center;">
                        <span style="font-size: 0.72rem; color: #8c786e; text-transform: uppercase; font-weight: 700;">Projected ROI</span>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #ff6b4a;">{custom_sim['roi_pct']:.1f}%</div>
                    </div>
                    """)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # SECTION 3: Standardized Action Packages
        # ---------------------------------------------------------
        st.subheader("Standardized Enterprise Retention Packages")
        packages = generate_prescriptive_packages(target_customer, champion_pipeline)

        pkg_cols = st.columns(len(packages))
        for idx, pkg in enumerate(packages):
            m = pkg["metrics"]
            with pkg_cols[idx]:
                actions_li = "".join([f"<li style='margin-bottom: 0.25rem;'>{act}</li>" for act in pkg['actions']])
                render_html(f"""
                <div class="package-card">
                    <div class="badge-soft badge-coral" style="margin-bottom: 0.5rem;">{pkg['target_persona']}</div>
                    <h3 style="font-size: 1.15rem; color: #1e1713; margin: 0 0 0.5rem 0;">{pkg['name']}</h3>
                    <p style="color: #6d5b52; font-size: 0.84rem; line-height: 1.5; min-height: 52px; margin-bottom: 1rem;">
                        {pkg['description']}
                    </p>
                    
                    <div style="background: #fff8f4; border: 1px solid #fedecd; border-radius: 10px; padding: 0.85rem; margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem; font-size: 0.82rem;">
                            <span style="color: #7c6a60;">Risk Transition:</span>
                            <strong style="color: #0da371;">{m['baseline_probability']*100:.1f}% &rarr; {m['intervened_probability']*100:.1f}% (-{m['risk_drop_pct']:.1f}%)</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem; font-size: 0.82rem;">
                            <span style="color: #7c6a60;">Annual Spend Cost:</span>
                            <strong style="color: #1e1713;">${m['annual_intervention_cost']:.2f}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem; font-size: 0.82rem;">
                            <span style="color: #7c6a60;">Preserved Revenue:</span>
                            <strong style="color: #1e1713;">${m['annual_revenue_saved']:.2f}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.82rem;">
                            <span style="color: #7c6a60;">Projected ROI:</span>
                            <strong style="color: #ff6b4a; font-size: 0.9rem;">{m['roi_pct']:.1f}%</strong>
                        </div>
                    </div>

                    <div style="font-size: 0.8rem; color: #4a3b34;">
                        <strong>Execution Protocol:</strong>
                        <ul style="padding-left: 1.15rem; margin-top: 0.4rem; color: #6d5b52;">
                            {actions_li}
                        </ul>
                    </div>
                </div>
                """)


# =========================================================
# PAGE 4: Portfolio Cohort Matrix
# =========================================================
elif page_selection == "Portfolio Cohort Matrix":
    render_html("""
    <div class="card-hero-peach">
        <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #e76f51; margin-bottom: 0.35rem;">
            Batch Scoring & Resource Allocation
        </div>
        <h1 style="font-size: 2.3rem; margin: 0; color: #1a120e;">
            Portfolio Cohort Matrix & Budget Optimizer
        </h1>
        <p style="color: #6d5a52; font-size: 1rem; margin-top: 0.4rem; max-width: 900px; line-height: 1.5;">
            Score subscriber cohorts to partition accounts into the 4-Quadrant Strategic Resource Allocation Matrix. 
            Optimizes retention team capacity by separating high-touch targets from automated and cost-sensitive segments.
        </p>
    </div>
    """)

    if champion_pipeline is None:
        st.error("Champion model not loaded.")
    else:
        col_up1, col_up2 = st.columns([1.5, 1])
        with col_up1:
            uploaded_file = st.file_uploader("Upload CSV for Batch Scoring", type=["csv"])
        with col_up2:
            st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
            load_benchmark = st.button("Load Standard Evaluation Benchmark Cohort (200 Records)")

        batch_source = None
        if uploaded_file is not None:
            batch_source = pd.read_csv(uploaded_file)
            st.caption(f"Loaded {len(batch_source):,} uploaded records.")
        elif load_benchmark or not df_sample.empty:
            batch_source = df_sample.head(200).copy()
            st.caption("Loaded 200 benchmark enterprise subscriber records.")

        if batch_source is not None:
            cohort_analysis = analyze_cohort_prescriptive_matrix(batch_source, champion_pipeline)
            df_scored = cohort_analysis["scored_dataframe"]
            fin = cohort_analysis["financial_summary"]
            q_counts = cohort_analysis["quadrant_counts"]

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Cohort Financial Projections & Program ROI")

            col_cf1, col_cf2, col_cf3, col_cf4 = st.columns(4)
            with col_cf1:
                render_html(render_metric_card(
                    "Annual Revenue at Risk",
                    f"${fin['annual_revenue_exposure']:,.0f}",
                    "Total cohort exposure",
                    badge_text="Gross Risk",
                    badge_type="coral"
                ))
            with col_cf2:
                render_html(render_metric_card(
                    "Recoverable Revenue",
                    f"${fin['total_recoverable_revenue']:,.0f}",
                    "Preserved via intervention",
                    badge_text="Target Protection",
                    badge_type="emerald"
                ))
            with col_cf3:
                render_html(render_metric_card(
                    "Retention Budget Needed",
                    f"${fin['required_retention_budget']:,.0f}",
                    "High-touch & automated spend",
                    badge_text="Investment",
                    badge_type="blue"
                ))
            with col_cf4:
                render_html(render_metric_card(
                    "Cohort Program ROI",
                    f"{fin['program_roi_pct']:.1f}%",
                    f"Net: ${fin['net_economic_benefit']:,.0f}",
                    badge_text="Net Multiple",
                    badge_type="emerald"
                ))

            st.markdown("<br>", unsafe_allow_html=True)

            # Interactive Plotly Matrix Scatter & Quadrant breakdown
            col_sc1, col_sc2 = st.columns([1.5, 1])
            with col_sc1:
                render_html("""
                <div style="margin-bottom: 0.5rem;">
                    <h3 style="margin: 0; font-size: 1.15rem;">Strategic Resource Allocation Matrix</h3>
                    <span style="font-size: 0.82rem; color: #7c6a60;">Quadrant map partitioning monthly charges against predicted churn risk</span>
                </div>
                """)
                fig_matrix = px.scatter(
                    df_scored,
                    x="MonthlyCharges",
                    y="predicted_churn_risk",
                    color="retention_quadrant",
                    color_discrete_map={
                        "Immediate High-Touch Retention": "#ff6b4a",
                        "Automated Self-Serve Incentives": "#f4a261",
                        "Cost-Sensitive Monitoring": "#84a98c",
                        "Secure Base (Maintain)": "#0da371"
                    },
                    hover_data=["Contract", "tenure"],
                    height=380,
                    labels={"predicted_churn_risk": "Predicted Churn Propensity", "MonthlyCharges": "Monthly Charges ($)"}
                )
                fig_matrix.add_hline(y=0.60, line_dash="dash", line_color="#d7ccc8")
                fig_matrix.add_vline(x=70.0, line_dash="dash", line_color="#d7ccc8")
                fig_matrix.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="#f3e3da", tickfont=dict(color='#261914')),
                    yaxis=dict(gridcolor="#f3e3da", tickfont=dict(color='#261914')),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="center", x=0.5, title=None, font=dict(color='#261914'))
                )
                st.plotly_chart(fig_matrix, width="stretch")

            with col_sc2:
                render_html("""
                <div style="margin-bottom: 0.5rem;">
                    <h3 style="margin: 0; font-size: 1.15rem;">Quadrant Volume & Strategy</h3>
                    <span style="font-size: 0.82rem; color: #7c6a60;">Account distribution & recommended operational protocols</span>
                </div>
                """)
                for q_name, count in q_counts.items():
                    pct = (count / len(df_scored)) * 100.0
                    badge_cls = "badge-coral" if "High-Touch" in q_name else ("badge-amber" if "Automated" in q_name else ("badge-blue" if "Cost" in q_name else "badge-emerald"))
                    render_html(f"""
                    <div style="background: #ffffff; border: 1px solid #f6ded2; border-radius: 10px; padding: 0.75rem 0.9rem; margin-bottom: 0.5rem; box-shadow: 0 2px 8px rgba(231, 111, 81, 0.03);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="color: #1e1713; font-size: 0.85rem;">{q_name}</strong>
                            <span class="badge-soft {badge_cls}">{count:,} ({pct:.1f}%)</span>
                        </div>
                    </div>
                    """)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Scored Cohort Data Table")
            display_cols = ["predicted_churn_risk", "retention_quadrant", "recommended_strategy", "Contract", "MonthlyCharges", "tenure"]
            if "customerID" in df_scored.columns:
                display_cols = ["customerID"] + display_cols
            available_cols = [c for c in display_cols if c in df_scored.columns]
            st.dataframe(df_scored[available_cols].head(100), width="stretch")

            csv_export = df_scored.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export Scored & Prescribed Dataset (CSV)",
                data=csv_export,
                file_name="prescriptive_scored_cohort.csv",
                mime="text/csv",
                width="stretch"
            )


# =========================================================
# PAGE 5: Model Benchmarks & Data Governance
# =========================================================
elif page_selection == "Model Benchmarks & Data":
    render_html("""
    <div class="card-hero-peach">
        <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #e76f51; margin-bottom: 0.35rem;">
            Empirical Validation & Data Lake Architecture
        </div>
        <h1 style="font-size: 2.3rem; margin: 0; color: #1a120e;">
            Model Benchmarks & Data Governance
        </h1>
        <p style="color: #6d5a52; font-size: 1rem; margin-top: 0.4rem; max-width: 900px; line-height: 1.5;">
            Rigorous cross-validation benchmarks comparing Scikit-Learn and Apache Spark MLlib candidate architectures, 
            paired with simulated HDFS Parquet storage partitions and global SHAP interpretability.
        </p>
    </div>
    """)

    metrics_list = model_eval.get("metrics_summary", [])
    if metrics_list:
        m_df = pd.DataFrame(metrics_list)
        
        st.subheader("Model Performance Leaderboard")
        st.dataframe(
            m_df[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "PR_AUC"]],
            width="stretch"
        )

        render_html("""
        <div class="insight-callout-peach" style="margin-top: 1rem;">
            <strong style="color: #d84315;">Production Champion:</strong> Gradient Boosted Trees achieved the highest Area Under the ROC Curve (0.8464) and PR-AUC (0.6639). 
            Tree ensembles capture high-dimensional non-linear interactions (e.g. Fiber optic internet with electronic checks) that linear boundaries cannot isolate.
        </div>
        """)

        st.markdown("<br>", unsafe_allow_html=True)

        # Benchmark ROC & Confusion Matrix Figures
        st.subheader("Comparative Benchmark Visualizations")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_path = BASE_DIR / "reports" / "figures" / "roc_curves_comparison.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Benchmark ROC Curves Comparison")
            
        with col_m2:
            cm_path = BASE_DIR / "reports" / "figures" / "confusion_matrices_comparison.png"
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion Matrices Comparison")

        st.markdown("<br>", unsafe_allow_html=True)

        # Global SHAP Summary & Detailed Explanations
        st.subheader("Global Explainable AI (SHAP Macro Insights)")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            beeswarm_path = BASE_DIR / "reports" / "figures" / "shap_summary_beeswarm.png"
            if beeswarm_path.exists():
                st.image(str(beeswarm_path), caption="Global SHAP Beeswarm Distribution")
        with col_g2:
            bar_path = BASE_DIR / "reports" / "figures" / "shap_feature_importance_bar.png"
            if bar_path.exists():
                st.image(str(bar_path), caption="Mean |SHAP| Global Feature Importance")

        # Global SHAP Educational Guide Box
        render_html("""
        <div style="background: #ffffff; border: 1px solid #f6ded2; border-radius: 14px; padding: 1.4rem 1.6rem; margin-top: 1.25rem; box-shadow: 0 4px 18px rgba(231, 111, 81, 0.05);">
            <div style="font-size: 0.82rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; color: #e76f51; margin-bottom: 0.5rem;">
                Interpreting Global SHAP Explainability &mdash; Population-Level Churn Dynamics
            </div>
            <p style="font-size: 0.88rem; color: #4a3b34; line-height: 1.6; margin: 0 0 1rem 0;">
                While Local SHAP explains an individual customer's prediction, Global SHAP synthesizes the collective decision landscape across all 7,043 enterprise accounts. 
                It uncovers macroeconomic patterns and operational friction points driving subscriber flight across the company.
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.2rem;">
                <div style="background: #fff8f4; border: 1px solid #fedecd; border-radius: 12px; padding: 1.15rem;">
                    <div style="font-size: 0.95rem; font-weight: 800; color: #ff6b4a; margin-bottom: 0.4rem;">
                        1. Global Beeswarm Distribution Plot
                    </div>
                    <ul style="padding-left: 1.15rem; margin: 0; font-size: 0.84rem; color: #5a463e; line-height: 1.6;">
                        <li style="margin-bottom: 0.35rem;"><strong>One Dot = One Subscriber:</strong> Every dot represents one of the 7,043 accounts in the dataset.</li>
                        <li style="margin-bottom: 0.35rem;"><strong>Horizontal Position (X-Axis):</strong> Dots to the right of zero pushed churn risk HIGHER for that customer; dots to the left pulled risk LOWER.</li>
                        <li style="margin-bottom: 0.35rem;"><strong>Dot Color (Feature Value):</strong> Red dots indicate high values of that feature (e.g., high monthly bill, long tenure); Blue dots indicate low values.</li>
                        <li style="margin-bottom: 0.35rem;"><strong>Cluster Thickness:</strong> Bulges in the swarm indicate high customer density at that specific impact level.</li>
                        <li><strong>Strategic Insight:</strong> Notice how red dots (high charges) cluster heavily on the right side of the plot&mdash;demonstrating that expensive monthly bills universally accelerate flight risk.</li>
                    </ul>
                </div>
                <div style="background: #f0fdf9; border: 1px solid #bbf7d0; border-radius: 12px; padding: 1.15rem;">
                    <div style="font-size: 0.95rem; font-weight: 800; color: #0da371; margin-bottom: 0.4rem;">
                        2. Mean Absolute |SHAP| Global Feature Ranking
                    </div>
                    <ul style="padding-left: 1.15rem; margin: 0; font-size: 0.84rem; color: #2e4a3d; line-height: 1.6;">
                        <li style="margin-bottom: 0.35rem;"><strong>Mathematical Magnitude:</strong> Measures the mean absolute impact <code>E[|SHAP|]</code> across the entire customer base.</li>
                        <li style="margin-bottom: 0.35rem;"><strong>Top-to-Bottom Importance:</strong> Features at the top have the largest global influence on whether accounts stay or churn.</li>
                        <li style="margin-bottom: 0.35rem;"><strong>Unbiased vs Traditional Gini:</strong> Unlike standard Random Forest feature importance (which unfairly favors high-cardinality continuous numbers), SHAP guarantees mathematical consistency and efficiency.</li>
                        <li><strong>Strategic Insight:</strong> Contract terms, Monthly Charges, and Fiber Optic Internet are the top 3 structural catalysts of churn across the entire enterprise.</li>
                    </ul>
                </div>
            </div>
        </div>
        """)

        st.markdown("<br>", unsafe_allow_html=True)

        # Storage & Schema Tab
        st.subheader("Distributed Storage & Schema Governance")
        tab_storage, tab_schema = st.tabs(["HDFS Parquet Lake Architecture", "Feature Data Dictionary"])
        
        with tab_storage:
            st.code("""
data/hdfs_simulation/warehouse/telco/
|-- Contract=Month-to-month/
|   `-- part-00000.parquet
|-- Contract=One year/
|   `-- part-00001.parquet
`-- Contract=Two year/
    `-- part-00002.parquet
            """, language="text")
            st.caption("Partitioned by contract commitment to optimize predicate pushdown and column projection in Spark queries.")

        with tab_schema:
            if not df_sample.empty:
                dtype_df = pd.DataFrame({
                    "Attribute Column": df_sample.columns,
                    "Data Type": [str(t) for t in df_sample.dtypes],
                    "Distinct Values": [df_sample[c].nunique() for c in df_sample.columns],
                    "Null Count": [df_sample[c].isna().sum() for c in df_sample.columns]
                })
                st.dataframe(dtype_df, width="stretch")

    else:
        st.warning("Model evaluation metrics not found.")
