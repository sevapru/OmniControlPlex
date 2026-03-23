"""
OmniControl Deck -- Gradio UI for OmniControlPlex
Port 4567

Accepts text prompt + per-joint spatial control signals, runs sample.generate,
and displays the resulting motion MP4 and allows .npy download.
"""
import os
import sys
import glob
import time
import subprocess
import tempfile
import shutil
import gradio as gr

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("MODELS_DIR", "/models/omnicontrolplex")
OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", "/outputs")

# Joint index map (OmniControl uses HumanML3D joint ordering)
JOINT_MAP = {
    "pelvis (0) — root position": 0,
    "left_hip (1)": 1,
    "right_hip (2)": 2,
    "spine1 (3)": 3,
    "left_knee (5)": 5,
    "right_knee (6)": 6,
    "left_ankle (10)": 10,
    "right_ankle (11)": 11,
    "left_foot (12)": 12,
    "right_foot (13)": 13,
    "neck (14)": 14,
    "head (15)": 15,
    "left_shoulder (16)": 16,
    "right_shoulder (17)": 17,
    "left_elbow (18)": 18,
    "right_elbow (19)": 19,
    "left_wrist (20)": 20,
    "right_wrist (21)": 21,
}

COND_MODES = {
    "text + spatial": "both_text_spatial",
    "spatial only": "only_spatial",
    "text only": "only_text",
}


def list_checkpoints() -> list[str]:
    patterns = [
        f"{MODELS_DIR}/**/*.pt",
        f"{MODELS_DIR}/*.pt",
        f"{WORKSPACE}/save/**/*.pt",
    ]
    ckpts = []
    for p in patterns:
        ckpts.extend(glob.glob(p, recursive=True))
    if not ckpts:
        return ["(no checkpoint found — place .pt file in /models/omnicontrolplex)"]
    return sorted(ckpts)


def generate_motion(
    text_prompt: str,
    model_path: str,
    control_joint: str,
    density: int,
    cond_mode: str,
    num_repetitions: int,
    guidance_param: float,
    motion_length: float,
    seed: int,
) -> tuple:
    if not text_prompt.strip() and cond_mode != "spatial only":
        return None, None, "Error: text prompt is required unless using spatial-only mode."
    if not model_path or not os.path.exists(model_path):
        return None, None, f"Error: checkpoint not found: {model_path}"

    joint_idx = JOINT_MAP.get(control_joint, 0)
    mode_val = COND_MODES.get(cond_mode, "both_text_spatial")

    run_id = f"run_{int(time.time())}"
    out_dir = os.path.join(OUTPUTS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "sample.generate",
        "--model_path", model_path,
        "--output_dir", out_dir,
        "--text_prompt", text_prompt if text_prompt.strip() else "predefined",
        "--control_joint", str(joint_idx),
        "--density", str(density),
        "--cond_mode", mode_val,
        "--num_repetitions", str(num_repetitions),
        "--guidance_param", str(guidance_param),
        "--motion_length", str(motion_length),
        "--seed", str(seed),
        "--num_samples", "1",
        "--batch_size", "1",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = WORKSPACE

    status_lines = [f"Running: {' '.join(cmd)}\n"]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=WORKSPACE, env=env
    )

    status_lines.append(result.stdout[-3000:] if result.stdout else "")
    if result.returncode != 0:
        status_lines.append(f"\nSTDERR:\n{result.stderr[-2000:]}")
        return None, None, "\n".join(status_lines)

    # Find outputs (generate.py puts them in out_dir or a subdirectory)
    mp4_files = sorted(glob.glob(f"{out_dir}/**/*.mp4", recursive=True))
    npy_files = sorted(glob.glob(f"{out_dir}/**/results.npy", recursive=True))

    mp4 = mp4_files[0] if mp4_files else None
    npy = npy_files[0] if npy_files else None

    status_lines.append(f"\nOutputs saved to: {out_dir}")
    if not mp4:
        status_lines.append("Warning: no MP4 found in output directory.")

    return mp4, npy, "\n".join(status_lines)


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="OmniControl Deck", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# OmniControl Deck\n"
        "**Joint-level spatial control for humanoid motion generation**  \n"
        "Specify a text prompt and which joint to spatially guide. "
        "Model: [OmniControlPlex](https://github.com/neu-vi/omnicontrol) on PyTorch 2.8 / CUDA 13."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            text_prompt = gr.Textbox(
                label="Motion description",
                placeholder="a person walks forward",
                lines=2,
            )
            model_path = gr.Dropdown(
                label="Checkpoint",
                choices=list_checkpoints(),
                value=list_checkpoints()[0],
                allow_custom_value=True,
            )
            control_joint = gr.Dropdown(
                label="Control joint",
                choices=list(JOINT_MAP.keys()),
                value="pelvis (0) — root position",
            )
            density = gr.Slider(
                0, 100, value=100, step=5,
                label="Spatial control density (%)",
                info="100% = control every frame; 0% = no spatial constraint",
            )
            cond_mode = gr.Radio(
                label="Conditioning mode",
                choices=list(COND_MODES.keys()),
                value="text + spatial",
            )

            with gr.Accordion("Advanced", open=False):
                num_repetitions = gr.Slider(1, 5, value=1, step=1, label="Repetitions")
                guidance_param = gr.Slider(1.0, 5.0, value=2.5, step=0.5, label="Guidance scale")
                motion_length = gr.Slider(2.0, 9.8, value=6.0, step=0.2, label="Motion length (s)")
                seed = gr.Number(value=10, label="Seed", precision=0)

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Output")
            video_out = gr.Video(label="Generated motion (stick figure)")
            npy_out = gr.File(label="Download results.npy")
            status_out = gr.Textbox(label="Status / log", lines=8, max_lines=20)

    refresh_btn = gr.Button("Refresh checkpoint list", size="sm")
    refresh_btn.click(
        lambda: gr.Dropdown(choices=list_checkpoints()),
        outputs=[model_path],
    )

    generate_btn.click(
        generate_motion,
        inputs=[
            text_prompt, model_path, control_joint, density, cond_mode,
            num_repetitions, guidance_param, motion_length, seed,
        ],
        outputs=[video_out, npy_out, status_out],
    )

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--server_name", default="0.0.0.0")
    p.add_argument("--server_port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 4567)))
    args = p.parse_args()
    demo.launch(server_name=args.server_name, server_port=args.server_port)
