"""
Network-Aware Adaptive Image Deblurring Application.

Combines:
1. Deep Learning Branch for Motion Blur:
   - Local Lightweight CNN (CPU, resource-constrained edge execution)
   - Server-based Restormer (Original & Custom-Loss on GPU)
   - Adaptive Decision Engine (Network-aware edge vs cloud computation)
2. Classical Digital Image Processing (DIP) Branch for Defocus Blur:
   - Circular Point Spread Function (PSF) generation
   - Wiener Deconvolution (Frequency domain)
   - Richardson-Lucy Deconvolution (Iterative spatial domain)
   - Quantitative evaluation (MSE, PSNR, SSIM) and error difference maps
"""

import io
import os
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

# Configure path so local modules and restoration package are found
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
PROJECT_ROOT = REPO_DIR.parent

for p in [str(BASE_DIR), str(REPO_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib
from lightweight.model import LightweightDeblurCNN
import restoration.defocus.psf as r_psf
import restoration.defocus.blur as r_blur
import restoration.defocus.wiener as r_wiener
import restoration.defocus.richardson_lucy as r_rl

importlib.reload(r_psf)
importlib.reload(r_blur)
importlib.reload(r_wiener)
importlib.reload(r_rl)

from restoration.defocus.psf import circular_psf
from restoration.defocus.blur import apply_defocus_blur
from restoration.defocus.wiener import wiener_deconv
from restoration.defocus.richardson_lucy import richardson_lucy_deconv

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================

st.set_page_config(
    page_title="Adaptive Image Deblurring Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme Polish
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222b;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        border-left: 4px solid #4a90e2;
    }
    .badge-edge {
        background-color: #27ae60;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .badge-cloud {
        background-color: #2980b9;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .badge-classical {
        background-color: #8e44ad;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://127.0.0.1:8000"
API_DEBLUR_URL = f"{API_BASE_URL}/deblur"
API_INFO_URL = f"{API_BASE_URL}/info"

LOCAL_CHECKPOINT = BASE_DIR / "lightweight" / "checkpoints_continued" / "lightweight_continued_best.pth"
CPU_DEVICE = torch.device("cpu")

LOCAL_PARAMETERS = 94243
RESTORMER_PARAMETERS = 26126644

# ==========================================
# HELPER FUNCTIONS & MODEL CACHING
# ==========================================

def get_file_size_mb(path):
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / (1024 ** 2)
    return None


@st.cache_resource
def load_local_model():
    """Load lightweight CNN onto CPU."""
    model = LightweightDeblurCNN(channels=32, num_blocks=5).to(CPU_DEVICE)
    if LOCAL_CHECKPOINT.exists():
        checkpoint = torch.load(LOCAL_CHECKPOINT, map_location=CPU_DEVICE, weights_only=False)
        sd = checkpoint.get("params", checkpoint.get("model_state_dict", checkpoint))
        model.load_state_dict(sd)
    model.eval()
    return model


def check_server_status():
    """Ping FastAPI server to check availability and GPU details."""
    try:
        resp = requests.get(API_INFO_URL, timeout=1.0)
        if resp.status_code == 200:
            return True, resp.json()
    except Exception:
        pass
    return False, {}


# ==========================================
# INFERENCE PIPELINES
# ==========================================

def process_motion_local(image: Image.Image):
    """Run Lightweight CNN on local CPU."""
    model = load_local_model()
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(CPU_DEVICE)

    start = time.perf_counter()
    with torch.inference_mode():
        restored = model(tensor)
    elapsed = time.perf_counter() - start

    restored = torch.clamp(restored, 0.0, 1.0).squeeze(0).cpu()
    output = transforms.ToPILImage()(restored)
    return output, elapsed


def process_motion_server(image: Image.Image, model_type: str = "original"):
    """Send image to GPU Restormer server via HTTP with resilient error parsing."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    start = time.perf_counter()
    try:
        response = requests.post(
            f"{API_DEBLUR_URL}?model_type={model_type}",
            files={"file": ("image.png", buf.getvalue(), "image/png")},
            timeout=180
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Cannot reach GPU server at {API_DEBLUR_URL}: {str(exc)}")

    request_time = time.perf_counter() - start

    if response.status_code != 200:
        error_msg = response.text
        try:
            error_msg = response.json().get("detail", error_msg)
        except Exception:
            pass
        raise RuntimeError(f"GPU Server Error ({response.status_code}): {error_msg}")

    gpu_inf_time = float(response.headers.get("X-Inference-Time", 0.0))
    restored = Image.open(io.BytesIO(response.content)).convert("RGB")
    return restored, request_time, gpu_inf_time


def apply_optional_edge_boost(image: Image.Image, enabled: bool, percent: int = 120):
    if not enabled or percent <= 0:
        return image
    return image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=percent, threshold=2))


# ==========================================
# IMAGE QUALITY METRICS
# ==========================================

def calculate_metrics_standard(prediction: Image.Image, ground_truth: Image.Image):
    """
    Standardized MSE, PSNR, SSIM calculation on float64 [0, 1] range.
    Ensures absolute mathematical parity across all models and scripts.
    """
    pred_np = np.asarray(prediction, dtype=np.float64) / 255.0
    gt_np = np.asarray(ground_truth, dtype=np.float64) / 255.0

    if pred_np.shape != gt_np.shape:
        raise ValueError(
            f"Dimension mismatch: Restored {pred_np.shape[:2]} vs Ground Truth {gt_np.shape[:2]}"
        )

    mse = float(np.mean((pred_np - gt_np) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    ssim_val = float(ssim(gt_np, pred_np, channel_axis=2, data_range=1.0))
    return mse, psnr, ssim_val


def render_metrics_columns(pred_img, gt_img, label="Quality Metrics"):
    if gt_img is None:
        st.info("💡 Upload a ground truth image to compute quantitative MSE, PSNR, and SSIM metrics.")
        return

    try:
        mse, psnr, ssim_val = calculate_metrics_standard(pred_img, gt_img)
        c1, c2, c3 = st.columns(3)
        c1.metric("MSE ↓", f"{mse:.6f}")
        c2.metric("PSNR ↑", f"{psnr:.2f} dB")
        c3.metric("SSIM ↑", f"{ssim_val:.4f}")
    except ValueError as e:
        st.warning(str(e))


# ==========================================
# SIDEBAR NAVIGATION & TELEMETRY
# ==========================================

st.sidebar.title("🔬 Deblurring Suite")
st.sidebar.markdown("**Network-Aware Adaptive Restoration**")
st.sidebar.divider()

branch = st.sidebar.radio(
    "Select Degradation Branch:",
    [
        "1. Motion Blur (Deep Learning)",
        "2. Defocus Blur (Classical DIP)"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("System Hardware & Telemetry")

server_online, server_telemetry = check_server_status()

if server_online:
    st.sidebar.success("● Cloud/Server: Online")
    st.sidebar.caption(f"**GPU:** {server_telemetry.get('gpu_name', 'Active')}")
    st.sidebar.caption(f"**VRAM Allocated:** {server_telemetry.get('vram_allocated_mb', 0)} MB / {server_telemetry.get('vram_total_mb', 0)} MB")
else:
    st.sidebar.error("○ Cloud/Server: Offline")
    st.sidebar.caption("Run `python -m uvicorn server:app` to activate GPU server.")

st.sidebar.info("● Local Edge: Available (CPU)")


# =========================================================================
# BRANCH 1: MOTION BLUR (DEEP LEARNING)
# =========================================================================
if branch.startswith("1."):
    st.title("Motion Blur Restoration (Deep Learning)")
    st.markdown(
        """
        Compare **edge-optimized lightweight inference** against **server-side high-capacity deep restoration**.
        Features an **Adaptive Decision Engine** that automatically determines whether to compute locally or offload to cloud.
        """
    )
    st.divider()

    # Model specifications summary card
    col_spec1, col_spec2 = st.columns(2)
    with col_spec1:
        st.markdown("**Edge Client: Lightweight CNN**")
        st.caption(f"Parameters: **{LOCAL_PARAMETERS:,}** | Checkpoint: **{get_file_size_mb(LOCAL_CHECKPOINT):.2f} MB** | Device: **CPU**")
    with col_spec2:
        st.markdown("**Cloud Server: Restormer Transformer**")
        st.caption(f"Parameters: **{RESTORMER_PARAMETERS:,}** | Checkpoint: **~99.85 MB** | Device: **GPU (CUDA)**")

    st.write("")

    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        motion_file = st.file_uploader("Upload Blurred Motion Image", type=["png", "jpg", "jpeg"], key="motion_input")
    with col_upload2:
        motion_gt_file = st.file_uploader("Upload Matching Ground Truth (Optional)", type=["png", "jpg", "jpeg"], key="motion_gt")

    if motion_file:
        input_image = Image.open(motion_file).convert("RGB")
        gt_image = Image.open(motion_gt_file).convert("RGB") if motion_gt_file else None

        st.subheader("Input Image Preview")
        st.image(input_image, caption=f"Input Motion Blurred Image ({input_image.width} × {input_image.height} px)", width=480)

        st.divider()
        st.subheader("Restoration Mode")

        mode = st.radio(
            "Choose Execution Strategy:",
            [
                "Local Processing (Edge CPU)",
                "Server Processing (Cloud GPU)",
                "Compare Both (Tradeoff Benchmark)",
                "Adaptive Decision Engine (Network-Aware)"
            ],
            horizontal=True
        )

        server_model_choice = "original"
        adaptive_priority = "Auto (Balanced)"

        if mode in ["Server Processing (Cloud GPU)", "Compare Both (Tradeoff Benchmark)"]:
            server_model_choice = st.selectbox(
                "Select Server Model Variant:",
                ["original", "custom_loss"],
                format_func=lambda x: "Original Restormer (Pretrained SOTA)" if x == "original" else "Fine-Tuned Restormer (Custom Loss)"
            )

        if mode == "Adaptive Decision Engine (Network-Aware)":
            st.info(
                "🤖 **Adaptive Policy Logic:** Dynamically balances computational latency, network payload, and restoration quality. "
                "Small images (< 0.25 MP) or offline states route locally to avoid network transmission penalties; "
                "larger images and quality-critical requests route to the GPU Restormer server when available."
            )
            adaptive_priority = st.selectbox(
                "Optimization Priority:",
                ["Auto (Balanced)", "Prioritize Low Latency (Edge Preference)", "Prioritize Maximum Restoration Quality (Cloud Preference)"]
            )

        with st.expander("⚡ Advanced Clarity & Edge Boost Controls", expanded=False):
            col_enh1, col_enh2 = st.columns(2)
            with col_enh1:
                passes = st.slider(
                    "Deblurring Passes (Iterative Refinement)",
                    min_value=1,
                    max_value=2,
                    value=1,
                    help="For large motion streaks (e.g. panning or high speed), feeding the output through a 2nd pass can resolve residual blur."
                )
            with col_enh2:
                enable_sharpening = st.checkbox(
                    "Apply High-Frequency Edge Definition (Unsharp Mask)",
                    value=False,
                    help="Restores punchy edge acutance and micro-contrast lost during motion smearing."
                )
                sharpen_strength = st.slider("Sharpening Intensity (%)", 50, 200, 130) if enable_sharpening else 0

        if st.button("Execute Restoration", type="primary"):
            # Determine routing for adaptive mode
            chosen_route = mode
            route_reason = ""

            if mode == "Adaptive Decision Engine (Network-Aware)":
                pixel_count = input_image.width * input_image.height
                if not server_online:
                    chosen_route = "Local Processing (Edge CPU)"
                    route_reason = "Server unavailable; seamlessly fallen back to Local Edge CPU."
                elif adaptive_priority == "Prioritize Low Latency (Edge Preference)":
                    chosen_route = "Local Processing (Edge CPU)"
                    route_reason = "Latency priority requested; executing directly on Local CPU without network serialization."
                elif adaptive_priority == "Prioritize Maximum Restoration Quality (Cloud Preference)":
                    chosen_route = "Server Processing (Cloud GPU)"
                    route_reason = "Quality priority requested; routing to high-capacity Restormer server."
                else: # Auto
                    if pixel_count < 250000:
                        chosen_route = "Local Processing (Edge CPU)"
                        route_reason = f"Image size ({input_image.width}x{input_image.height} = {pixel_count:,} px) is compact; edge processing avoids round-trip HTTP overhead."
                    else:
                        chosen_route = "Server Processing (Cloud GPU)"
                        route_reason = f"Image resolution ({input_image.width}x{input_image.height}) benefits significantly from transformer attention; routing to GPU Restormer."

                st.success(f"🎯 **Decision:** Routed to **{chosen_route}**\n\n*Rationale:* {route_reason}")

            # Execution logic
            if chosen_route == "Local Processing (Edge CPU)":
                with st.spinner("Processing on Local CPU with Lightweight CNN..."):
                    out_img, elapsed = process_motion_local(input_image)
                    if passes == 2:
                        out_img, elapsed2 = process_motion_local(out_img)
                        elapsed += elapsed2
                    out_img = apply_optional_edge_boost(out_img, enable_sharpening, sharpen_strength)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(input_image, caption="Blurred Input", use_container_width=True)
                with c2:
                    st.image(out_img, caption=f"Local Restored Output ({passes}-Pass{' + Edge Boost' if enable_sharpening else ''})", use_container_width=True)

                st.metric("CPU Execution Time", f"{elapsed:.4f} seconds")
                render_metrics_columns(out_img, gt_image)

            elif chosen_route == "Server Processing (Cloud GPU)":
                if not server_online:
                    st.error("Server is offline! Please start the FastAPI server via `python -m uvicorn server:app --host 127.0.0.1 --port 8000`.")
                else:
                    try:
                        with st.spinner(f"Sending image to GPU server ({server_model_choice})..."):
                            out_img, total_time, gpu_time = process_motion_server(input_image, server_model_choice)
                            if passes == 2:
                                out_img, total_time2, gpu_time2 = process_motion_server(out_img, server_model_choice)
                                total_time += total_time2
                                gpu_time += gpu_time2
                            out_img = apply_optional_edge_boost(out_img, enable_sharpening, sharpen_strength)

                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(input_image, caption="Blurred Input", use_container_width=True)
                        with c2:
                            st.image(out_img, caption=f"Server Restored Output ({server_model_choice}, {passes}-Pass{' + Edge Boost' if enable_sharpening else ''})", use_container_width=True)

                        c_time1, c_time2 = st.columns(2)
                        c_time1.metric("Pure GPU Inference Time", f"{gpu_time:.4f} s")
                        c_time2.metric("Total Request Round-Trip Time", f"{total_time:.4f} s")
                        render_metrics_columns(out_img, gt_image)
                    except Exception as exc:
                        st.error(f"❌ Restoration failed: {exc}")

            elif chosen_route == "Compare Both (Tradeoff Benchmark)":
                if not server_online:
                    st.error("GPU Server is currently offline. Start the server to perform side-by-side comparison.")
                else:
                    with st.spinner("Executing Local Lightweight CNN..."):
                        local_out, local_time = process_motion_local(input_image)
                        if passes == 2:
                            local_out, l_time2 = process_motion_local(local_out)
                            local_time += l_time2
                        local_out = apply_optional_edge_boost(local_out, enable_sharpening, sharpen_strength)

                    server_success = False
                    try:
                        with st.spinner(f"Executing Cloud Restormer ({server_model_choice})..."):
                            server_out, total_server_time, gpu_time = process_motion_server(input_image, server_model_choice)
                            if passes == 2:
                                server_out, s_time2, g_time2 = process_motion_server(server_out, server_model_choice)
                                total_server_time += s_time2
                                gpu_time += g_time2
                            server_out = apply_optional_edge_boost(server_out, enable_sharpening, sharpen_strength)
                            server_success = True
                    except Exception as exc:
                        st.error(f"❌ Cloud GPU execution failed: {exc}")

                    st.subheader("Side-by-Side Visual Comparison")
                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.markdown("**Edge Lightweight CNN (Local CPU)**")
                        st.image(local_out, use_container_width=True)
                        st.metric("CPU Inference Time", f"{local_time:.4f} s")
                        render_metrics_columns(local_out, gt_image)

                    with col_r:
                        if server_success:
                            st.markdown(f"**Cloud Restormer ({server_model_choice} on GPU)**")
                            st.image(server_out, use_container_width=True)
                            st.metric("Server Request Time (GPU: {:.2f}s)".format(gpu_time), f"{total_server_time:.4f} s")
                            render_metrics_columns(server_out, gt_image)
                        else:
                            st.warning("Server comparison skipped due to error above.")

                    st.divider()
                    st.subheader("Summary Analysis & Tradeoff")
                    st.markdown(f"""
                    | Dimension | Edge Lightweight CNN | Cloud Restormer ({server_model_choice}) |
                    | :--- | :--- | :--- |
                    | **Parameter Count** | **94,243** (~277x fewer) | **26,126,644** |
                    | **Model Checkpoint Size** | **0.37 MB** | **99.85 MB** |
                    | **Computational Target** | Resource-constrained edge CPU | High-performance GPU Server |
                    | **Observed Latency** | **{local_time:.3f} s** | **{total_server_time:.3f} s** (GPU compute: {gpu_time:.3f} s) |
                    """)


# =========================================================================
# BRANCH 2: DEFOCUS BLUR (CLASSICAL DIGITAL IMAGE PROCESSING)
# =========================================================================
elif branch.startswith("2."):
    st.title("Defocus Blur Restoration (Classical DIP)")
    st.markdown(
        """
        Explore classical image deconvolution for defocus blur:
        - **Circular Point Spread Function (PSF):** Models circular optical lens aperture.
        - **Wiener Deconvolution:** Frequency-domain linear restoration balancing inverse filtering and noise amplification.
        - **Richardson-Lucy Deconvolution:** Iterative spatial-domain maximum likelihood restoration preserving non-negativity.
        """
    )
    st.divider()

    defocus_mode = st.radio(
        "Select Experiment Mode:",
        [
            "Interactive Synthetic Blur & Restoration (Ground Truth Benchmark)",
            "Pre-Blurred Image Restoration"
        ],
        horizontal=True
    )

    if defocus_mode.startswith("Interactive"):
        st.subheader("1. Reference Image & Degradation Controls")
        sharp_upload = st.file_uploader("Upload Sharp Reference Image (or leave empty to use built-in sample)", type=["png", "jpg", "jpeg"], key="sharp_input")

        if sharp_upload:
            base_image = Image.open(sharp_upload).convert("RGB")
        else:
            default_sample_path = PROJECT_ROOT / "Restormer" / "Motion_Deblurring" / "Datasets" / "test" / "GoPro" / "ground_truth" / "image1.jpg"
            if default_sample_path.exists():
                base_image = Image.open(default_sample_path).convert("RGB")
                st.caption(f"Using default sharp reference: `image1.jpg` ({base_image.width} × {base_image.height})")
            else:
                base_image = Image.new("RGB", (400, 300), color=(120, 140, 180))

        c_deg1, c_deg2 = st.columns(2)
        with c_deg1:
            blur_radius = st.slider("Circular PSF Blur Radius (R in pixels)", min_value=1.0, max_value=8.0, value=3.0, step=0.5)
        with c_deg2:
            noise_sigma = st.slider("Additive Gaussian Noise Std (σ relative to [0, 1])", min_value=0.0, max_value=0.02, value=0.002, step=0.001, format="%.3f")

        # Synthesize blur
        sharp_np = np.asarray(base_image, dtype=np.float64) / 255.0
        psf = circular_psf(radius=blur_radius)
        degraded_np = apply_defocus_blur(sharp_np, psf, noise_sigma=noise_sigma, seed=42)
        degraded_img = Image.fromarray((degraded_np * 255.0).round().astype(np.uint8))

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.image(base_image, caption="Original Sharp Ground Truth", use_container_width=True)
        with col_p2:
            st.image(degraded_img, caption=f"Degraded Defocus Blur (R={blur_radius}px, σ={noise_sigma})", use_container_width=True)

        st.divider()
        st.subheader("2. Classical Restoration Parameters")

        col_alg1, col_alg2 = st.columns(2)
        with col_alg1:
            st.markdown("**Wiener Filter Settings**")
            reg_k = st.select_slider(
                "Noise-to-Signal Regularizer (K)",
                options=[1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2, 5e-2],
                value=3e-3,
                format_func=lambda x: f"{x:.1e} ({'Crisp Sharp' if x <= 3e-3 else 'Smooth'})"
            )
            wiener_smooth = st.slider("High-Frequency Grain Suppression (Tikhonov)", min_value=0.0, max_value=0.05, value=0.005, step=0.005, format="%.3f", help="Damps noise amplification at circular PSF frequency zeroes.")
            wiener_acutance = st.slider("⚡ Edge Acutance / Sharpening Boost (%)", min_value=0, max_value=150, value=80, step=10, help="Sharpens reconstructed edges to restore razor-sharp details.")
            wiener_denoise = st.checkbox("Extra Smooth Denoise (Optional for Heavy Noise)", value=False, help="Leave off for maximum sharpness. Turn on only if input has heavy grain.")
        with col_alg2:
            st.markdown("**Richardson-Lucy Settings**")
            rl_iters = st.slider("EM Iterations (N)", min_value=15, max_value=60, value=35, step=5, help="Controls deconvolution depth (35-40 is optimal).")
            rl_damping = st.slider("Noise Damping (White's Damping)", min_value=0.0, max_value=0.01, value=0.003, step=0.001, format="%.3f", help="Prevents noise amplification in smooth regions where residuals fall below noise floor.")
            rl_acutance = st.slider("⚡ Edge Acutance Boost (%)", min_value=0, max_value=150, value=80, step=10, help="Sharpens reconstructed high-frequency boundaries.")

        if st.button("Run Classical Deconvolution", type="primary"):
            # 1. Wiener
            t0 = time.perf_counter()
            wiener_np = wiener_deconv(degraded_np, psf, reg_k=reg_k, smoothness_weight=wiener_smooth, denoise_post=wiener_denoise)
            t_wiener = time.perf_counter() - t0
            wiener_img = Image.fromarray((wiener_np * 255.0).round().astype(np.uint8))
            if wiener_acutance > 0:
                wiener_img = wiener_img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=wiener_acutance, threshold=2))

            # 2. Richardson-Lucy
            t0 = time.perf_counter()
            rl_np = richardson_lucy_deconv(degraded_np, psf, num_iters=rl_iters, damping_threshold=rl_damping, edge_boost=rl_acutance)
            t_rl = time.perf_counter() - t0
            rl_img = Image.fromarray((rl_np * 255.0).round().astype(np.uint8))

            st.subheader("Restoration Results")
            c_res1, c_res2 = st.columns(2)

            with c_res1:
                st.markdown(f"**Wiener Deconvolution** *(K={reg_k:.1e})*")
                st.image(wiener_img, use_container_width=True)
                st.metric("Execution Time", f"{t_wiener:.4f} s")
                render_metrics_columns(wiener_img, base_image)

            with c_res2:
                st.markdown(f"**Richardson-Lucy** *({rl_iters} iterations)*")
                st.image(rl_img, use_container_width=True)
                st.metric("Execution Time", f"{t_rl:.4f} s")
                render_metrics_columns(rl_img, base_image)

            st.divider()
            st.subheader("Absolute Error Maps (|Restored - Ground Truth| × 4)")
            err_w = np.clip(np.abs(wiener_np - sharp_np) * 4.0, 0, 1)
            err_rl = np.clip(np.abs(rl_np - sharp_np) * 4.0, 0, 1)

            c_err1, c_err2 = st.columns(2)
            c_err1.image(err_w, caption="Wiener Residual Error", use_container_width=True)
            c_err2.image(err_rl, caption="Richardson-Lucy Residual Error", use_container_width=True)

            st.markdown(f"""
            > **DIP Method Comparison:**
            > - **Wiener Filtering:** Direct frequency-domain division $F = \\frac{{H^*}}{{|H|^2 + K}} G$. Extremely fast ({t_wiener:.3f}s), but ringing artifacts can occur if $K$ is set too small under noise.
            > - **Richardson-Lucy:** Iterative maximum-likelihood deconvolution ($f^{{(t+1)}} = f^{{(t)}} \\cdot [\\frac{{g}}{{f^{{(t)}} * h}} * h]$). Preserves non-negativity and boundary details better under noise, but takes {t_rl:.3f}s for {rl_iters} iterations.
            """)

    else: # Pre-blurred mode
        st.subheader("Restore Pre-Blurred Defocus Image")
        blurry_upload = st.file_uploader("Upload Blurry Defocus Image", type=["png", "jpg", "jpeg"], key="pre_blurred")
        blurry_gt = st.file_uploader("Upload Matching Ground Truth (Optional)", type=["png", "jpg", "jpeg"], key="pre_gt")

        if blurry_upload:
            b_img = Image.open(blurry_upload).convert("RGB")
            b_gt = Image.open(blurry_gt).convert("RGB") if blurry_gt else None

            st.image(b_img, caption="Uploaded Blurry Image", width=480)

            c_p1, c_p2 = st.columns(2)
            with c_p1:
                est_radius = st.slider("Estimated Blur Radius (R)", 1.0, 8.0, 3.0, 0.5)
                pre_acutance = st.slider("⚡ Edge Acutance Boost (%)", 0, 150, 80, 10, help="Sharpens reconstructed edges.")
            with c_p2:
                classical_method = st.selectbox("Deconvolution Method:", ["Wiener Deconvolution", "Richardson-Lucy", "Compare Both"])
                pre_reg_k = st.select_slider(
                    "Wiener Regularizer (K)",
                    options=[1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2, 5e-2],
                    value=3e-3,
                    format_func=lambda x: f"{x:.1e} ({'Crisp Sharp' if x <= 3e-3 else 'Smooth'})"
                )

            if st.button("Restore Image", type="primary"):
                b_np = np.asarray(b_img, dtype=np.float64) / 255.0
                psf_est = circular_psf(radius=est_radius)

                if classical_method in ["Wiener Deconvolution", "Compare Both"]:
                    t0 = time.perf_counter()
                    w_np = wiener_deconv(b_np, psf_est, reg_k=pre_reg_k, smoothness_weight=0.005, denoise_post=False)
                    t_w = time.perf_counter() - t0
                    w_img = Image.fromarray((w_np * 255.0).round().astype(np.uint8))
                    if pre_acutance > 0:
                        w_img = w_img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=pre_acutance, threshold=2))

                    st.markdown("**Wiener Deconvolution Output**")
                    st.image(w_img, width=540)
                    st.metric("Time", f"{t_w:.4f} s")
                    render_metrics_columns(w_img, b_gt)

                if classical_method in ["Richardson-Lucy", "Compare Both"]:
                    t0 = time.perf_counter()
                    rl_np = richardson_lucy_deconv(b_np, psf_est, num_iters=35, damping_threshold=0.003, edge_boost=pre_acutance)
                    t_rl = time.perf_counter() - t0
                    rl_img = Image.fromarray((rl_np * 255.0).round().astype(np.uint8))

                    st.markdown("**Richardson-Lucy Output (25 iters)**")
                    st.image(rl_img, width=540)
                    st.metric("Time", f"{t_rl:.4f} s")
                    render_metrics_columns(rl_img, b_gt)


# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption(
    "Network-Aware Adaptive Image Deblurring | BTech Final Project | "
    "Deep Learning (Restormer & Lightweight CNN) + Digital Image Processing (Wiener & Richardson-Lucy)"
)
