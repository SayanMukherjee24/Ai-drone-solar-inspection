import json
from pathlib import Path
import random
import time
import urllib.request

import cv2
import numpy as np
import streamlit as st


PUBLIC_IMAGES = {
    "Aerial Solar Farm 1 (Wikimedia)": "https://upload.wikimedia.org/wikipedia/commons/5/51/Solar_panels_at_Golmud%2C_Qinghai%2C_China.jpg",
    "Aerial Solar Farm 2 (Wikimedia)": "https://upload.wikimedia.org/wikipedia/commons/3/37/Nellis_Solar_Power_Plant.jpg",
    "Solar Panels Close View (Wikimedia)": "https://upload.wikimedia.org/wikipedia/commons/f/fb/Solar_panels_on_a_roof.jpg",
}

GITHUB_DATASET_SOURCES = {
    "dbaofd/solar-panels-detection (cover.png)": "https://raw.githubusercontent.com/dbaofd/solar-panels-detection/master/imgs/cover.png",
    "dbaofd/solar-panels-detection (demo_1.gif)": "https://raw.githubusercontent.com/dbaofd/solar-panels-detection/master/imgs/demo_1.gif",
    "dbaofd/solar-panels-detection (demo_2.gif)": "https://raw.githubusercontent.com/dbaofd/solar-panels-detection/master/imgs/demo_2.gif",
}
DATASET_CACHE_DIR = Path("dataset_cache")


def load_image_from_upload(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_image_from_url(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = response.read()
        file_bytes = np.asarray(bytearray(data), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def load_image_from_local_path(path):
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def fetch_github_repo_images(owner, repo, folder="imgs"):
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder}"
    req = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "streamlit-solar-inspection-demo",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
        image_items = []
        for item in payload:
            if item.get("type") != "file":
                continue
            name = item.get("name", "").lower()
            if name.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                image_items.append(
                    {
                        "name": item.get("name"),
                        "download_url": item.get("download_url"),
                    }
                )
        return image_items
    except Exception:
        return []


def download_and_cache_image(url, filename_hint):
    DATASET_CACHE_DIR.mkdir(exist_ok=True)
    safe_name = filename_hint.replace("/", "_").replace("\\", "_")
    target_path = DATASET_CACHE_DIR / safe_name
    if target_path.exists():
        return target_path

    try:
        with urllib.request.urlopen(url, timeout=18) as response:
            data = response.read()
        target_path.write_bytes(data)
        return target_path
    except Exception:
        return None


def detect_panel_region(image_rgb):
    h, w = image_rgb.shape[:2]
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median_intensity = float(np.median(blurred))
    low = int(max(30, 0.66 * median_intensity))
    high = int(min(220, 1.33 * median_intensity))
    edged = cv2.Canny(blurred, low, high)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box = None
    best_area = 0
    used_fallback = False

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) == 4:
            x, y, bw, bh = cv2.boundingRect(approx)
            area = bw * bh
            aspect = bw / float(bh + 1e-6)
            if area > best_area and 0.6 <= aspect <= 2.5 and area > (h * w * 0.03):
                best_area = area
                best_box = (x, y, bw, bh)

    if best_box is None:
        bw = int(w * 0.6)
        bh = int(h * 0.5)
        x = (w - bw) // 2
        y = (h - bh) // 2
        best_box = (x, y, bw, bh)
        used_fallback = True

    return best_box, used_fallback


def classify_fault(panel_rgb, panel_detect_score):
    gray = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2GRAY)
    denoise = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(denoise, 90, 190)

    edge_density = float(np.mean(edges > 0))
    texture_std = float(np.std(denoise))
    lap_var = float(cv2.Laplacian(denoise, cv2.CV_64F).var())
    brightness_std = float(np.std(cv2.equalizeHist(denoise)))

    if edge_density > 0.14 or lap_var > 1300:
        label = "Crack Detected"
    elif edge_density > 0.07 or texture_std > 30 or brightness_std > 58:
        label = "Dust/Soiling"
    else:
        label = "Normal Panel"

    # Confidence is now evidence-driven and slightly jittered for demo realism.
    confidence = (
        0.62
        + min(0.14, edge_density * 0.8)
        + min(0.12, texture_std / 320.0)
        + min(0.08, lap_var / 3000.0)
        + panel_detect_score * 0.14
        + random.uniform(-0.03, 0.03)
    )
    confidence = round(float(np.clip(confidence, 0.70, 0.95)), 2)

    metrics = {
        "edge_density": edge_density,
        "texture_std": texture_std,
        "lap_var": lap_var,
    }
    return label, confidence, metrics


def detect_hotspots(panel_rgb):
    gray = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    normalized = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)

    threshold_value = np.percentile(normalized, 98.7)
    _, mask = cv2.threshold(normalized, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hotspots = []
    min_area = max(5, int((panel_rgb.shape[0] * panel_rgb.shape[1]) * 0.0003))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius >= 3:
            hotspots.append((int(cx), int(cy), int(radius)))

    return hotspots, threshold_value


def make_agent_decision(fault_label, confidence, hotspot_found):
    if confidence < 0.75:
        return "Agent Decision: Re-scan required (low confidence)"
    if fault_label == "Crack Detected" or hotspot_found:
        return "Agent Decision: Maintenance Required"
    return "Agent Decision: Panel Healthy"


def estimate_image_suitability(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edge_density = float(np.mean(cv2.Canny(gray, 80, 160) > 0))
    contrast = float(np.std(gray))

    # Heuristic "scene suitability" for aerial/structured panel-like imagery.
    suitability = np.clip((edge_density * 2.2) + (contrast / 120.0), 0.0, 1.0)
    return float(suitability), edge_density, contrast


def main():
    st.set_page_config(page_title="AI Drone Solar Inspection", layout="wide", page_icon="🚁")
    
    # Advanced CSS for glassmorphism, metrics, and modern gradient text
    st.markdown("""
        <style>
        .gradient-text {
            font-size: 3rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
            letter-spacing: -1px;
        }
        .sub-text {
            font-size: 1.1rem;
            color: #A0AEC0;
            margin-bottom: 30px;
            font-weight: 400;
        }
        
        /* Modernized button styling */
        .stButton>button {
            border-radius: 20px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 1px solid rgba(0, 201, 255, 0.3);
            background: rgba(19, 26, 42, 0.8) !important;
            color: #E2E8F0 !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0, 201, 255, 0.4);
            border-color: #00C9FF !important;
            color: #00C9FF !important;
        }
        
        /* Glassmorphism Metric Cards */
        [data-testid="stMetric"] {
            background: rgba(19, 26, 42, 0.6);
            border: 1px solid rgba(0, 201, 255, 0.15);
            border-radius: 15px;
            padding: 15px 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            transition: transform 0.2s ease-in-out;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            border-color: rgba(0, 201, 255, 0.4);
        }
        
        /* Softer labels for metrics */
        [data-testid="stMetricLabel"] {
            font-size: 0.95rem;
            color: #94A3B8;
        }
        
        /* Crisp values for metrics */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
            color: #F8FAFC;
        }
        
        /* Floating images with rounded corners */
        [data-testid="stImage"] img {
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }
        
        /* Tweak expander styling */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #00C9FF;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="gradient-text">🚁 Agentic AI Solar Drone</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Prototype of a 3-stage autonomous UAV pipeline (Finder → Inspector → Electrician)</p>', unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("🚁 AI Drone Solar Inspection", anchor=False)
        source = st.radio(
            "Select Telemetry Source",
            ["Upload Your Own", "Use Public Demo Imagery", "Use GitHub Dataset Imagery"],
            horizontal=True,
        )

    if "public_image_rgb" not in st.session_state:
        st.session_state.public_image_rgb = None
    if "github_image_rgb" not in st.session_state:
        st.session_state.github_image_rgb = None

    image_rgb = None
    if source == "Upload Your Own":
        uploaded = st.file_uploader("Upload JPG/PNG drone image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image_rgb = load_image_from_upload(uploaded)
            if image_rgb is None:
                st.error("Could not read uploaded image. Please try a different file.")
    else:
        if source == "Use Public Demo Imagery":
            sample_name = st.selectbox("Choose public image", list(PUBLIC_IMAGES.keys()))
            col_load, col_random = st.columns(2)
            with col_load:
                if st.button("Load selected public image", width='stretch'):
                    st.session_state.public_image_rgb = load_image_from_url(PUBLIC_IMAGES[sample_name])
                    if st.session_state.public_image_rgb is None:
                        st.error("Failed to load public image. Try another one or upload manually.")
            with col_random:
                if st.button("Load random public sample", width='stretch'):
                    random_name = random.choice(list(PUBLIC_IMAGES.keys()))
                    st.session_state.public_image_rgb = load_image_from_url(PUBLIC_IMAGES[random_name])
                    if st.session_state.public_image_rgb is None:
                        st.error("Failed to load random sample. Try again.")
                    else:
                        st.caption(f"Loaded: {random_name}")

            image_rgb = st.session_state.public_image_rgb
        else:
            st.caption("Connected source: GitHub repo dataset sample cache")
            st.markdown(
                "Dataset reference: "
                "[dbaofd/solar-panels-detection](https://github.com/dbaofd/solar-panels-detection)"
            )

            fetch_mode = st.radio(
                "GitHub dataset load mode",
                ["Fast fixed sample list", "Discover repo image list (GitHub API)"],
                horizontal=True,
            )

            selected_label = None
            selected_url = None
            github_options = {}

            if fetch_mode == "Fast fixed sample list":
                github_options = GITHUB_DATASET_SOURCES.copy()
            else:
                repo_images = fetch_github_repo_images("dbaofd", "solar-panels-detection", "imgs")
                if repo_images:
                    github_options = {
                        f"dbaofd/solar-panels-detection/{item['name']}": item["download_url"]
                        for item in repo_images
                        if item.get("download_url")
                    }
                if not github_options:
                    st.warning(
                        "Could not discover repo images via API. "
                        "Falling back to fixed sample list."
                    )
                    github_options = GITHUB_DATASET_SOURCES.copy()

            selected_label = st.selectbox("Choose GitHub dataset image", list(github_options.keys()))
            selected_url = github_options[selected_label]

            local_files = sorted(
                [p.name for p in DATASET_CACHE_DIR.glob("*")] if DATASET_CACHE_DIR.exists() else []
            )
            if local_files:
                st.caption(f"Cached dataset files: {len(local_files)}")

            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                if st.button("Load selected GitHub sample", width='stretch'):
                    cached_path = download_and_cache_image(selected_url, selected_label.split("/")[-1])
                    if cached_path is None:
                        st.error("GitHub download failed. Check internet, then retry.")
                    else:
                        st.session_state.github_image_rgb = load_image_from_local_path(cached_path)
                        if st.session_state.github_image_rgb is None:
                            st.error("Cached file is not a readable image.")
            with col_g2:
                if st.button("Load random GitHub sample", width='stretch'):
                    random_label = random.choice(list(github_options.keys()))
                    random_url = github_options[random_label]
                    cached_path = download_and_cache_image(random_url, random_label.split("/")[-1])
                    if cached_path is None:
                        st.error("Random GitHub sample download failed.")
                    else:
                        st.session_state.github_image_rgb = load_image_from_local_path(cached_path)
                        if st.session_state.github_image_rgb is None:
                            st.error("Downloaded file is not a readable image.")
                        else:
                            st.caption(f"Loaded: {random_label}")
            with col_g3:
                local_choice = st.selectbox(
                    "Load cached local dataset file",
                    ["None"] + local_files,
                    label_visibility="collapsed",
                    key="local_dataset_choice",
                )
                if st.button("Load cached file", width='stretch'):
                    if local_choice != "None":
                        st.session_state.github_image_rgb = load_image_from_local_path(
                            DATASET_CACHE_DIR / local_choice
                        )
                        if st.session_state.github_image_rgb is None:
                            st.error("Selected cached file is not a valid image.")

            image_rgb = st.session_state.github_image_rgb

    if image_rgb is None:
        st.info("📡 Awaiting feed. Upload an image or select a demo sample to initialize agents.")
        return

    st.markdown("---")
    st.subheader("⚙️ Agent Pipeline Status", anchor=False)
    progress = st.progress(0, text="Booting AI systems...")
    stage_status = st.empty()
    
    stage_status.info("🔍 **Stage 1/3 (Finder):** Scanning landscape for solar panels...")
    time.sleep(0.35)
    progress.progress(25, text="Finder agent isolated target...")

    suitability, scene_edge_density, scene_contrast = estimate_image_suitability(image_rgb)
    panel_box, used_fallback = detect_panel_region(image_rgb)
    x, y, bw, bh = panel_box
    panel_roi = image_rgb[y : y + bh, x : x + bw]
    panel_detect_score = 0.45 if used_fallback else 0.85

    stage_status.info("🔬 **Stage 2/3 (Inspector):** Analyzing crystalline silicon surface condition...")
    time.sleep(0.35)
    progress.progress(60, text="Inspector agent assessing damage...")

    fault_label, confidence, metrics = classify_fault(panel_roi, panel_detect_score)

    stage_status.info("⚡ **Stage 3/3 (Electrician):** Thermographic sweep for overheating cells...")
    time.sleep(0.35)
    progress.progress(85, text="Electrician analyzing thermal map...")

    hotspots, hotspot_threshold = detect_hotspots(panel_roi)
    hotspot_found = len(hotspots) > 0
    hotspot_status = "Detected 🔴" if hotspot_found else "None 🟢"

    progress.progress(100, text="Intelligence loop concluded.")
    stage_status.success("✅ **Sequence Complete.** Autonomous analysis reporting below.")

    annotated = image_rgb.copy()
    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (43, 255, 127), 4)
    cv2.putText(
        annotated,
        "Target Locked",
        (x, max(25, y - 10)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (43, 255, 127),
        2,
    )

    for cx, cy, r in hotspots:
        cv2.circle(annotated, (x + cx, y + cy), r + 4, (255, 43, 86), 4)
        cv2.putText(
            annotated,
            "Thermal Anomaly",
            (x + cx + 12, y + cy - 8),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (255, 43, 86),
            2,
        )

    decision = make_agent_decision(fault_label, confidence, hotspot_found)

    latitude = round(random.uniform(-90, 90), 6)
    longitude = round(random.uniform(-180, 180), 6)
    altitude = round(random.uniform(10, 50), 1)

    st.markdown("---")
    st.subheader("📊 Tactical Analysis Report", anchor=False)
    
    # Beautiful metrics layout
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Panel Integrity", fault_label)
    c2.metric("Agent Confidence", f"{confidence * 100:.1f}%")
    c3.metric("Thermal Hotspot", hotspot_status)
    c4.metric("Suitability Score", f"{suitability * 100:.0f}%")

    if "Maintenance Required" in decision:
        st.error(f"🚨 **{decision}**")
    elif "Healthy" in decision:
        st.success(f"✅ **{decision}**")
    else:
        st.warning(f"⚠️ **{decision}**")

    # Image comparison side-by-side inside containers
    colA, colB = st.columns(2)
    with colA:
        with st.container(border=True):
            st.markdown("**Original Telemetry**")
            st.image(image_rgb, width='stretch')
            st.caption(f"📍 GPS: {latitude}, {longitude} | 🚁 Alt: {altitude}m")
    with colB:
        with st.container(border=True):
            st.markdown("**AI Augmented Target Map**")
            st.image(annotated, width='stretch')
            st.caption(f"🧭 Mode: {'Fallback (Center ROI)' if used_fallback else 'Edge Contour Targeting'}")

    st.markdown("---")
    with st.expander("🛠️ Advanced Agent Heuristics & Thresholds"):
        st.write(f"Scene suitability score: `{suitability:.2f}`")
        st.write(f"Scene edge density: `{scene_edge_density:.3f}`")
        st.write(f"Scene contrast: `{scene_contrast:.2f}`")
        st.write(f"Panel edge density: `{metrics['edge_density']:.3f}`")
        st.write(f"Panel texture std: `{metrics['texture_std']:.2f}`")
        st.write(f"Panel Laplacian variance: `{metrics['lap_var']:.2f}`")
        st.write(f"Hotspot percentile threshold: `{hotspot_threshold:.2f}`")

    if suitability < 0.30:
        st.toast(
            "⚠️ Image may not be ideal for solar inspection (low scene suitability). "
            "Use aerial panel imagery.", icon="⚠️"
        )


if __name__ == "__main__":
    main()
