"""Sentinel-1 monthly land-cover research dashboard.

Run with:
    streamlit run app.py

Earth Engine access must be authenticated separately in the environment running
Streamlit. All output maps and statistics are model predictions, NOT ground truth.
"""

import io
import os
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import segmentation_models_pytorch as smp
import streamlit as st
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from pyproj import Transformer
from rasterio.io import MemoryFile


st.set_page_config(
    page_title="Sentinel-1 | Monthly LULC Monitoring",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS = PROJECT_DIR / "unet_sentinel1_finetune_best.pth"
EE_PROJECT = os.environ.get("EE_PROJECT", "mydataportfolio-484121")

CLASSES = [
    "Trees", "Shrubland", "Grassland", "Cropland", "Built-up", "Bare land",
    "Snow/Ice", "Water", "Wetland", "Mangroves", "Moss/Lichen",
]
COLORS = [
    "#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000", "#b4b4b4",
    "#f0f0f0", "#0064c8", "#0096a0", "#00cf75", "#fae6a0",
]
COLOR_MAP = ListedColormap(COLORS)
COLOR_NORM = BoundaryNorm(np.arange(-0.5, 11.5, 1), COLOR_MAP.N)
MAIN_CLASSES = ["Trees", "Grassland", "Cropland", "Built-up", "Water"]
PRESETS = {
    "Heidelberg": (49.3988, 8.6724),
    "Stuttgart": (48.7758, 9.1829),
    "Mannheim": (49.4875, 8.4660),
    "Karlsruhe": (49.0069, 8.4037),
    "Freiburg": (47.9990, 7.8421),
    "Koblenz": (50.3643, 7.6056),
    "Ulm": (48.4011, 9.9876),
    "Tübingen": (48.5216, 9.0576),
    "Passau": (48.57356, 13.47636),
}


# Light, polished interface. Styling never modifies the actual model outputs.
st.markdown(
    """
    <style>
    .stApp { background: #f7f9fc; color: #173052; }
    [data-testid="stSidebar"] { background: #eef3fb; border-right: 1px solid #e3eaf6; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #18375f; }
    .block-container { padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1800px; }
    .hero-title { font-size: 2rem; font-weight: 850; color: #183b78; letter-spacing: -.04em; }
    .hero-desc { color: #53677f; margin-top: .15rem; margin-bottom: .75rem; }
    .section-title { font-size: 1.16rem; color: #173969; font-weight: 800; margin: .2rem 0 .75rem; }
    .muted-note { font-size: .86rem; color: #5c7189; }
    .class-key { display:flex; align-items:center; gap:.6rem; margin: .32rem 0; font-size:.9rem; }
    .key-swatch { width: 17px; height:17px; border:1px solid #d3dbe6; border-radius:4px; display:inline-block; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid #e4eaf3; border-radius: 10px; padding: 12px 15px; }
    div[data-testid="stVerticalBlockBorderWrapper"] > div { border-color:#e5ebf4; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model():
    """Initialize Earth Engine and load the saved fine-tuned U-Net on CPU."""
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Missing trained weights: {WEIGHTS.name}")
    ee.Initialize(project=EE_PROJECT)
    net = smp.Unet(
        encoder_name="resnet18", encoder_weights=None, in_channels=2, classes=11
    )
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    net.eval()
    return net


def predict_one_month(net, latitude, longitude, start_date, end_date):
    """Download one monthly SAR median in EPSG:3035 and infer one 256x256 LULC map."""
    tx = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    cx, cy = tx.transform(longitude, latitude)
    xmin, ymax = cx - 1280.0, cy + 1280.0
    roi = ee.Geometry.Rectangle(
        [cx - 1280, cy - 1280, cx + 1280, cy + 1280],
        proj="EPSG:3035", geodesic=False,
    )
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )
    scene_count = int(s1.size().getInfo())
    if scene_count == 0:
        raise ValueError(f"No matching Sentinel-1 observations for {start_date[:7]}")

    url = s1.median().getDownloadURL({
        "bands": ["VV", "VH"],
        "crs": "EPSG:3035",
        "crs_transform": [10, 0, xmin, 0, -10, ymax],
        "dimensions": [256, 256],
        "format": "GEO_TIFF",
    })
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with MemoryFile(response.content) as mem:
        with mem.open() as src:
            radar = src.read(masked=True)

    if radar.shape != (2, 256, 256):
        raise ValueError(f"Unexpected radar dimensions for {start_date[:7]}")
    if np.ma.getmaskarray(radar).any():
        raise ValueError(f"Invalid/missing radar pixels for {start_date[:7]}")
    radar = radar.data.astype(np.float32)
    if not np.isfinite(radar).all():
        raise ValueError(f"Non-finite radar values for {start_date[:7]}")

    # Exactly the VV/VH normalization used in training.
    vv = (np.clip(radar[0], -30.0, 5.0) + 30.0) / 35.0
    vh = (np.clip(radar[1], -35.0, 0.0) + 35.0) / 35.0
    image = torch.from_numpy(np.stack([vv, vh])[None].astype(np.float32))
    with torch.no_grad():
        pred = net(image).argmax(dim=1)[0].numpy().astype(np.uint8)

    # Radar false-colour (VV, VH, mean). NOT an optical satellite image.
    sar_rgb = np.clip(np.stack([vv, vh, (vv + vh) / 2], axis=-1), 0, 1)
    return pred, sar_rgb, scene_count


def figure_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=175, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def map_grid_figure(months, masks):
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), facecolor="white")
    for i, ax in enumerate(axes.flat):
        ax.imshow(masks[i], cmap=COLOR_MAP, norm=COLOR_NORM, interpolation="nearest")
        ax.set_title(months[i], fontsize=12, weight="bold", pad=8)
        ax.axis("off")
    legend_items = [Patch(facecolor=c, edgecolor="none", label=name)
                    for c, name in zip(COLORS, CLASSES)]
    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               fontsize=8.5, frameon=False, bbox_to_anchor=(.5, .01))
    fig.tight_layout(rect=[0, .09, 1, 1], h_pad=.8, w_pad=.5)
    return fig


def comparison_figure(month_a, month_b, radar_a, radar_b, pred_a, pred_b):
    gained = (pred_a != 4) & (pred_b == 4)
    lost = (pred_a == 4) & (pred_b != 4)
    change = np.full((256, 256, 3), 0.30, dtype=np.float32)
    change[gained] = (1.0, 0.84, 0.0)     # yellow
    change[lost] = (0.16, 0.65, 1.0)     # blue

    fig, ax = plt.subplots(2, 3, figsize=(13, 7.1), facecolor="white")
    ax[0, 0].imshow(radar_a)
    ax[0, 0].set_title(f"Sentinel-1 SAR · {month_a}", fontsize=10)
    ax[0, 1].imshow(radar_b)
    ax[0, 1].set_title(f"Sentinel-1 SAR · {month_b}", fontsize=10)
    ax[0, 2].imshow(change)
    ax[0, 2].set_title("Built-up gain / loss", fontsize=10)
    ax[1, 0].imshow(pred_a, cmap=COLOR_MAP, norm=COLOR_NORM, interpolation="nearest")
    ax[1, 0].set_title(f"LULC · {month_a}", fontsize=10)
    ax[1, 1].imshow(pred_b, cmap=COLOR_MAP, norm=COLOR_NORM, interpolation="nearest")
    ax[1, 1].set_title(f"LULC · {month_b}", fontsize=10)
    ax[1, 2].imshow(radar_b)
    overlay = np.zeros((256, 256, 4), dtype=np.float32)
    overlay[gained] = (1.0, .84, 0, .82)
    overlay[lost] = (.16, .65, 1, .82)
    ax[1, 2].imshow(overlay)
    ax[1, 2].set_title("Change overlay on current SAR", fontsize=10)
    for axis in ax.flat:
        axis.axis("off")
    fig.tight_layout(h_pad=1.0, w_pad=.6)
    return fig, gained, lost


def trend_figure(coverage, chosen_classes):
    fig, ax = plt.subplots(figsize=(8.5, 4.3), facecolor="white")
    indices = np.arange(len(coverage))
    for name in chosen_classes:
        idx = CLASSES.index(name)
        ax.plot(indices, coverage[name], marker="o", markersize=3,
                linewidth=2, label=name, color=COLORS[idx])
    ax.set_xticks(indices)
    ax.set_xticklabels(coverage["Month"], rotation=48, ha="right", fontsize=8)
    ax.set_ylabel("Predicted coverage (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("Monthly land-cover coverage", fontsize=12, weight="bold")
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def on_city_change():
    city = st.session_state["city_preset"]
    if city in PRESETS:
        st.session_state["latitude_input"], st.session_state["longitude_input"] = PRESETS[city]


# ---- Sidebar ---------------------------------------------------------------
if "latitude_input" not in st.session_state:
    st.session_state["latitude_input"] = 50.3643
if "longitude_input" not in st.session_state:
    st.session_state["longitude_input"] = 7.6056

with st.sidebar:
    st.markdown("### 🛰️ LULC Monitoring")
    st.caption("Sentinel-1 · U-Net")
    st.divider()
    st.markdown("### Location & time")
    st.selectbox("Suggested region", list(PRESETS.keys()) + ["Custom coordinates"],
                 key="city_preset", on_change=on_city_change)
    with st.form("run_form"):
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0,
                                   format="%.6f", key="latitude_input")
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0,
                                    format="%.6f", key="longitude_input")
        start_month = st.text_input("Starting month (YYYY-MM)", value="2021-01", disabled=True)
        run = st.form_submit_button("▶ Generate 12 Monthly Maps", use_container_width=True,
                                    type="primary")

    st.divider()
    st.markdown("### Land-cover classes")
    for name, color in zip(CLASSES, COLORS):
        st.markdown(
            f'<div class="class-key"><span class="key-swatch" style="background:{color}"></span>{name}</div>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.markdown("### About")
    st.caption("Semantic segmentation with Sentinel-1 VV/VH radar input only. "
               "2021 WorldCover labels were used for supervised training.")


# ---- Headline and model check ----------------------------------------------
st.markdown('<div class="hero-title">🛰️ Sentinel-1 Monthly LULC Monitoring</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-desc">Radar-only U-Net research prototype · 12 monthly maps, '
            'land-cover statistics and before/after comparisons</div>',
            unsafe_allow_html=True)

try:
    model = get_model()
except Exception as exc:
    st.error(f"Initialization failed: {exc}")
    st.info("In Colab, authenticate with ee.Authenticate() in the notebook, "
            "then restart the Streamlit server. Check that the fine-tuned "
            "model weights are saved alongside app.py.")
    st.stop()
else:
    st.success("U-Net model and Google Earth Engine connected")


# ---- Prediction (run only on user action) -----------------------------------
if run:
    try:
        if len(start_month) != 7 or start_month[4] != "-" or not (start_month[:4] + start_month[5:]).isdigit():
            raise ValueError("Enter a valid start month in YYYY-MM format, e.g. 2021-01")
        first = pd.Period(start_month, freq="M")
        periods = pd.period_range(first, periods=12, freq="M")
        if periods[-1].end_time > pd.Timestamp.now():
            raise ValueError("Choose a start month for which all 12 months have finished")

        months, maps, radar_images, rows, scenes = [], [], [], [], []
        progress = st.progress(0)
        status = st.empty()
        for i, p in enumerate(periods):
            month = str(p)
            status.caption(f"Retrieving Sentinel-1 and predicting {month} ({i+1}/12)")
            prediction, radar_rgb, scene_count = predict_one_month(
                model, latitude, longitude,
                p.start_time.strftime("%Y-%m-%d"),
                (p + 1).start_time.strftime("%Y-%m-%d"),
            )
            months.append(month)
            maps.append(prediction)
            radar_images.append(radar_rgb)
            scenes.append(scene_count)
            counts = np.bincount(prediction.ravel(), minlength=len(CLASSES))
            rows.append({"Month": month, **dict(zip(CLASSES, 100 * counts / counts.sum()))})
            progress.progress((i + 1) / 12)
        status.empty()
        progress.empty()
        st.session_state["lulc_results"] = {
            "months": months,
            "maps": maps,
            "radar_images": radar_images,
            "coverage": pd.DataFrame(rows),
            "scenes": scenes,
            "latitude": latitude,
            "longitude": longitude,
            "selected_region": st.session_state.get("city_preset", "Custom coordinates"),
        }
        st.toast("12 monthly predictions generated", icon="✅")
    except Exception as exc:
        st.error(f"Monthly processing failed: {exc}")
        st.caption("Any results from a prior successful run remain below, if available.")


# ---- Results (persist across widget reruns) --------------------------------
if "lulc_results" not in st.session_state:
    st.info("Choose a location in the sidebar and click **Generate 12 Monthly Maps**. "
            "The dashboard will display real outputs from your saved U-Net model.")
    st.warning("Research limitation: the model overpredicts Built-up land at some locations. "
               "Colourful predictions do not imply accuracy.")
    st.stop()

result = st.session_state["lulc_results"]
months = result["months"]
maps = result["maps"]
radar_images = result["radar_images"]
coverage = result["coverage"]
monthly_change = coverage.copy()
monthly_change[CLASSES] = coverage[CLASSES].diff()  # percentage points
first_to_last = pd.DataFrame({
    "Class": CLASSES,
    "First month (%)": [float(coverage.iloc[0][name]) for name in CLASSES],
    "Last month (%)": [float(coverage.iloc[-1][name]) for name in CLASSES],
    "Change (pp)": [float(coverage.iloc[-1][name] - coverage.iloc[0][name])
                    for name in CLASSES],
})

st.caption(
    f"Showing generated predictions for {result['latitude']:.6f}, "
    f"{result['longitude']:.6f} · {months[0]} to {months[-1]} · "
    f"{sum(result['scenes'])} Sentinel-1 scene observations across monthly composites"
)

kpis = st.columns(3)
for block, name in zip(kpis, ["Trees", "Built-up", "Water"]):
    with block:
        delta = coverage.iloc[-1][name] - coverage.iloc[0][name]
        st.metric(f"{name} · {months[-1]}", f"{coverage.iloc[-1][name]:.2f}%",
                  f"{delta:+.2f} pp vs {months[0]}")

# Top row: 12 maps + coverage/change tables
left, right = st.columns([1.65, 1], gap="large")
with left:
    st.markdown('<div class="section-title">🗺️ 12 monthly LULC maps</div>',
                unsafe_allow_html=True)
    fig_maps = map_grid_figure(months, maps)
    st.pyplot(fig_maps, use_container_width=True)
    maps_png = figure_png(fig_maps)
    plt.close(fig_maps)
    st.download_button("⬇ Download 12-map figure (PNG)", maps_png,
                       file_name="monthly_lulc_maps.png", mime="image/png")

with right:
    st.markdown('<div class="section-title">📋 Monthly land-cover coverage (%)</div>',
                unsafe_allow_html=True)
    st.dataframe(coverage[["Month"] + MAIN_CLASSES].round(2),
                 hide_index=True, use_container_width=True)
    with st.expander("All 11 land-cover classes"):
        st.dataframe(coverage.round(2), hide_index=True, use_container_width=True)
    st.download_button("⬇ Download coverage CSV", coverage.to_csv(index=False),
                       file_name="monthly_coverage.csv", mime="text/csv")
    st.markdown('<div class="section-title">📈 First-to-last change (percentage points)</div>',
                unsafe_allow_html=True)
    st.dataframe(first_to_last[first_to_last["Class"].isin(MAIN_CLASSES)].round(2),
                 hide_index=True, use_container_width=True)
    with st.expander("Full first-to-last change table"):
        st.dataframe(first_to_last.round(2), hide_index=True, use_container_width=True)
    st.download_button("⬇ Download total changes CSV", first_to_last.to_csv(index=False),
                       file_name="total_changes_pp.csv", mime="text/csv")

st.divider()

# Bottom row: before/after comparison + trend chart
left2, right2 = st.columns([1.65, 1], gap="large")
with left2:
    st.markdown('<div class="section-title">🔁 Previous month vs current month</div>',
                unsafe_allow_html=True)
    current_month = st.selectbox("Select current month", months[1:], index=0)
    idx = months.index(current_month)
    month_a, month_b = months[idx - 1], months[idx]
    fig_diff, gained, lost = comparison_figure(
        month_a, month_b,
        radar_images[idx - 1], radar_images[idx],
        maps[idx - 1], maps[idx],
    )
    st.pyplot(fig_diff, use_container_width=True)
    diff_png = figure_png(fig_diff)
    plt.close(fig_diff)
    st.download_button("⬇ Download before/after comparison (PNG)", diff_png,
                       file_name=f"comparison_{month_a}_to_{month_b}.png", mime="image/png")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Predicted Built-up gain", f"{100*gained.mean():.2f}%")
    with d2:
        st.metric("Predicted Built-up loss", f"{100*lost.mean():.2f}%")
    with d3:
        st.metric("Net change", f"{100*(gained.mean()-lost.mean()):+.2f} pp")
    st.caption("Yellow = newly predicted Built-up · Blue = no longer predicted Built-up · "
               "Grey = unchanged Built-up status. Radar panels are false-colour SAR, not optical photos.")

with right2:
    st.markdown('<div class="section-title">📊 Monthly LULC trend</div>',
                unsafe_allow_html=True)
    chosen = st.multiselect("Show classes", CLASSES,
                            default=["Trees", "Built-up", "Water"])
    if chosen:
        fig_trend = trend_figure(coverage, chosen)
        st.pyplot(fig_trend, use_container_width=True)
        trend_png = figure_png(fig_trend)
        plt.close(fig_trend)
        st.download_button("⬇ Download trend graph (PNG)", trend_png,
                           file_name="monthly_lulc_trend.png", mime="image/png")
    st.markdown('<div class="section-title">📐 Month-to-month changes (pp)</div>',
                unsafe_allow_html=True)
    st.dataframe(monthly_change[["Month"] + MAIN_CLASSES].round(2),
                 hide_index=True, use_container_width=True)
    with st.expander("All classes / complete change table"):
        st.dataframe(monthly_change.round(2), hide_index=True, use_container_width=True)
    st.download_button("⬇ Download monthly change CSV", monthly_change.to_csv(index=False),
                       file_name="monthly_changes_pp.csv", mime="text/csv")

st.warning(
    "**Research limitations:** These are predictions, not verified land-cover changes, "
    "individual building counts or construction dates. The model overpredicts Built-up "
    "land in the Stuttgart diagnostic. Changes between monthly SAR composites may reflect "
    "radar acquisition conditions and model errors rather than real changes."
)
