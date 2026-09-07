import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gradio as gr

from gui.model_runner import (
    load_curated,
    get_sample_metadata,
    get_random_sample,
    render_single_model,
    render_comparison_grid,
    truncate_description,
)

CURATED_RECOMMENDATIONS = load_curated()
PRESET_CHOICES = [f"[{item.get('split', 'test').upper()}] {item['uid']}" for item in CURATED_RECOMMENDATIONS]
LABEL_TO_UID = {f"[{item.get('split', 'test').upper()}] {item['uid']}": item["uid"] for item in CURATED_RECOMMENDATIONS}

DEFAULT_CHOICE = PRESET_CHOICES[0] if PRESET_CHOICES else ""
DEFAULT_UID = CURATED_RECOMMENDATIONS[0]["uid"] if CURATED_RECOMMENDATIONS else "0009/00097460"
DEFAULT_PROMPT = truncate_description(CURATED_RECOMMENDATIONS[0]["prompt"]) if CURATED_RECOMMENDATIONS else ""
DEFAULT_SPLIT_BADGE = "**Dataset Split:** 🏷️ `TEST`"

def on_preset_select(preset_label: str):
    uid = LABEL_TO_UID.get(preset_label, "")
    meta = get_sample_metadata(uid)
    split = meta.get("split", "test")
    prompt = truncate_description(meta.get("prompt", ""))
    badge = f"**Dataset Split:** 🏷️ `{split.upper()}`"
    return uid, prompt, badge

def on_random_click():
    uid, split, prompt = get_random_sample()
    badge = f"**Dataset Split:** 🏷️ `{split.upper()}`"
    return uid, prompt, badge

def on_search_click(uid: str):
    clean_uid = uid.strip()
    meta = get_sample_metadata(clean_uid)
    split = meta.get("split", "test")
    prompt = truncate_description(meta.get("prompt", ""))
    badge = f"**Dataset Split:** 🏷️ `{split.upper()}`"
    return prompt, badge

def build_app():
    with gr.Blocks(title="Text2CAD") as demo:
        gr.Markdown("# Text2CAD")

        with gr.Tabs():
            # ── TAB 1: PER MODEL RENDER (PREDICTED VS GROUND TRUTH) ──
            with gr.TabItem("Per Model Render"):
                with gr.Row():
                    with gr.Column(scale=4):
                        t1_model = gr.Dropdown(
                            label="Model Selection",
                            choices=["M3", "Baseline", "M1", "M2", "M4"],
                            value="M3",
                            interactive=True,
                        )
                        t1_preset = gr.Dropdown(
                            label="Instance Selection",
                            choices=PRESET_CHOICES,
                            value=DEFAULT_CHOICE,
                            interactive=True,
                        )

                        with gr.Row():
                            t1_random_btn = gr.Button("🎲 Random Sample", variant="secondary", scale=1)

                        with gr.Row():
                            t1_uid = gr.Textbox(
                                label="Target Sample UID",
                                value=DEFAULT_UID,
                                placeholder="e.g. 0009/00097460",
                                interactive=True,
                                scale=3,
                            )
                            t1_search_btn = gr.Button("🔍 Search by UID", variant="secondary", scale=1)

                        t1_split_badge = gr.Markdown(value=DEFAULT_SPLIT_BADGE)

                        t1_prompt = gr.Textbox(
                            label="CAD Description (Prompt)",
                            value=DEFAULT_PROMPT,
                            lines=2,
                            max_lines=3,
                            interactive=False,
                        )
                        t1_btn = gr.Button("Render", variant="primary")

                    with gr.Column(scale=8):
                        with gr.Tabs():
                            with gr.TabItem("3D Interactive View"):
                                with gr.Row():
                                    t1_pred_3d = gr.Model3D(label="Predicted Model Geometry (.stl)")
                                    t1_gt_3d = gr.Model3D(label="Ground Truth Geometry (.stl)")

                            with gr.TabItem("2D Isometric Render"):
                                with gr.Row():
                                    t1_pred_2d = gr.Image(label="Predicted Isometric Render (.png)", type="filepath")
                                    t1_gt_2d = gr.Image(label="Ground Truth Isometric Render (.png)", type="filepath")

                            with gr.TabItem("DualSeq Representation"):
                                with gr.Row():
                                    t1_pred_code = gr.Code(label="Predicted DualSeq Representation", language="yaml")
                                    t1_gt_code = gr.Code(label="Ground Truth DualSeq Representation", language="yaml")

            # ── TAB 2: ALL MODEL RENDER ───────
            with gr.TabItem("All Model Render"):
                with gr.Row():
                    with gr.Column(scale=4):
                        t2_preset = gr.Dropdown(
                            label="Instance Selection",
                            choices=PRESET_CHOICES,
                            value=DEFAULT_CHOICE,
                            interactive=True,
                        )

                        with gr.Row():
                            t2_random_btn = gr.Button("🎲 Random Sample", variant="secondary", scale=1)

                        with gr.Row():
                            t2_uid = gr.Textbox(
                                label="Target Sample UID",
                                value=DEFAULT_UID,
                                placeholder="e.g. 0009/00097460",
                                interactive=True,
                                scale=3,
                            )
                            t2_search_btn = gr.Button("🔍 Search by UID", variant="secondary", scale=1)

                        t2_split_badge = gr.Markdown(value=DEFAULT_SPLIT_BADGE)

                        t2_prompt = gr.Textbox(
                            label="CAD Description (Prompt)",
                            value=DEFAULT_PROMPT,
                            lines=2,
                            max_lines=3,
                            interactive=False,
                        )
                        t2_cd_table = gr.Markdown(
                            value="Select or render a sample to view Chamfer Distance (CD) scores."
                        )
                        t2_btn = gr.Button("Render", variant="primary")

                    with gr.Column(scale=8):
                        t2_grid = gr.Image(
                            label="All Model Comparison Grid",
                            type="filepath",
                        )

        # Tab 1 Event Bindings
        t1_preset.change(
            fn=on_preset_select,
            inputs=[t1_preset],
            outputs=[t1_uid, t1_prompt, t1_split_badge],
        )

        t1_random_btn.click(
            fn=on_random_click,
            inputs=[],
            outputs=[t1_uid, t1_prompt, t1_split_badge],
        )

        t1_search_btn.click(
            fn=on_search_click,
            inputs=[t1_uid],
            outputs=[t1_prompt, t1_split_badge],
        )

        t1_btn.click(
            fn=render_single_model,
            inputs=[t1_model, t1_uid],
            outputs=[
                t1_pred_3d,
                t1_gt_3d,
                t1_pred_2d,
                t1_gt_2d,
                t1_pred_code,
                t1_gt_code,
                t1_split_badge,
                t1_prompt,
            ],
        )

        # Tab 2 Event Bindings
        t2_preset.change(
            fn=on_preset_select,
            inputs=[t2_preset],
            outputs=[t2_uid, t2_prompt, t2_split_badge],
        )

        t2_random_btn.click(
            fn=on_random_click,
            inputs=[],
            outputs=[t2_uid, t2_prompt, t2_split_badge],
        )

        t2_search_btn.click(
            fn=on_search_click,
            inputs=[t2_uid],
            outputs=[t2_prompt, t2_split_badge],
        )

        t2_btn.click(
            fn=render_comparison_grid,
            inputs=[t2_uid],
            outputs=[
                t2_grid,
                t2_split_badge,
                t2_prompt,
                t2_cd_table,
            ],
        )

    return demo

if __name__ == "__main__":
    print("[GUI] Initializing Text2CAD Studio...", flush=True)
    app = build_app()
    print("[GUI] Launching Gradio server at http://localhost:7860 ...", flush=True)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=[
            os.path.join(PROJECT_ROOT, "out"),
            os.path.join(PROJECT_ROOT, "gui/assets"),
        ],
    )
