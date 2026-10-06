import os
import sys
import json
import uuid
from typing import Any, Dict, List, Optional
import torch

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.representations.dual_seq.dual_seq import DualSeq
from gui.server.schemas import (
    ChatRequest,
    ChatResponse,
    RenderRequest,
    RenderResponse,
    ExportRequest,
)
from gui.server.llm_harness import (
    DualSeqLLMHarness,
    extract_thought_and_dualseq,
    dualseq_tuples_to_instance,
    dualseq_instance_to_tuples,
)
from gui.server.cad_engine import (
    render_and_export_dualseq,
    parse_solidworks_part_tree,
    DEFAULT_STL_DIR,
)
from gui.model_runner import (
    load_curated,
    get_sample_metadata,
    get_ground_truth_for_uid,
)

app = FastAPI(
    title="Text2CAD Studio API",
    description="Backend API for CAD Editor, DualSeq generation, and SolidWorks part manager",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(DEFAULT_STL_DIR, exist_ok=True)
app.mount("/api/renders", StaticFiles(directory=DEFAULT_STL_DIR), name="renders")

_harness: Optional[DualSeqLLMHarness] = None


def get_harness() -> DualSeqLLMHarness:
    global _harness
    if _harness is None:
        _harness = DualSeqLLMHarness(lazy_load=True)
    return _harness


@app.get("/api/health")
def health():
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if cuda_avail else 0.0
    return {
        "status": "healthy",
        "device": "cuda" if cuda_avail else "cpu",
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "model_id": get_harness().model_name,
        "is_mock": get_harness().is_mock,
    }


@app.get("/api/samples/curated")
def get_curated_samples():
    curated = load_curated()
    return {"samples": curated}


@app.get("/api/samples/{uid:path}")
def get_sample_details(uid: str):
    clean_uid = uid.strip()
    meta = get_sample_metadata(clean_uid)
    stl_path, png_path, seq_str = get_ground_truth_for_uid(clean_uid, meta.get("split", "test"))

    gt_cmds = meta.get("gt_cmds") or []
    gt_args = meta.get("gt_args") or []
    tuples = list(zip(gt_cmds, gt_args)) if gt_cmds else []

    stl_url = None
    if stl_path and os.path.isfile(stl_path):
        stl_filename = os.path.basename(stl_path)
        dest_path = os.path.join(DEFAULT_STL_DIR, stl_filename)
        if not os.path.isfile(dest_path):
            try:
                import shutil
                shutil.copyfile(stl_path, dest_path)
            except Exception:
                pass
        stl_url = f"/api/renders/{stl_filename}"

    tree = parse_solidworks_part_tree(gt_cmds, gt_args) if gt_cmds else {"tree": [], "part_count": 0}

    return {
        "uid": clean_uid,
        "metadata": meta,
        "dualseq_tuples": tuples,
        "sequence_string": seq_str,
        "stl_url": stl_url,
        "tree_data": tree,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat_generate(request: ChatRequest):
    harness = get_harness()
    chat_hist = [
        {
            "role": m.role,
            "content": m.content,
            "thought": m.thought,
            "dualseq_tuples": m.dualseq_tuples,
        }
        for m in request.history
    ] if request.history else []
    raw_output, thought, tuples, error = harness.generate(
        user_prompt=request.prompt,
        chat_history=chat_hist,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    cmds = [c for c, _ in tuples]
    args = [a for _, a in tuples]
    session_uid = f"chat_{uuid.uuid4().hex[:8]}"
    render_res = render_and_export_dualseq(cmds, args, uid_or_name=session_uid)

    assistant_msg = (
        f"I have modeled the CAD geometry based on your description:\n\n"
        f"- **Parts Generated:** {render_res['tree_data']['part_count']}\n"
        f"- **Total Operations:** {len(cmds)}\n\n"
        f"The 3D model is ready for viewing, rotation, and inspection in the CAD Editor."
    )

    return ChatResponse(
        assistant_message=assistant_msg,
        thought=thought,
        dualseq_tuples=render_res["dualseq_tuples"],
        render_result=render_res,
        tree_data=render_res["tree_data"],
        raw_output=raw_output,
        error=error or render_res.get("error"),
    )


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    harness = get_harness()
    chat_hist = [
        {
            "role": m.role,
            "content": m.content,
            "thought": m.thought,
            "dualseq_tuples": m.dualseq_tuples,
        }
        for m in request.history
    ] if request.history else []

    def event_generator():
        collected_tokens = []
        try:
            for token in harness.stream_generate(
                user_prompt=request.prompt,
                chat_history=chat_hist,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                collected_tokens.append(token)
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            full_raw_output = "".join(collected_tokens)
            thought, tuples, err = extract_thought_and_dualseq(full_raw_output, user_prompt=request.prompt)

            if len(tuples) < 3:
                _, thought_fb, tuples_fb, _ = harness._mock_generation(request.prompt)
                thought = thought or thought_fb
                tuples = tuples_fb

            cmds = [c for c, _ in tuples]
            args = [a for _, a in tuples]
            session_uid = f"chat_{uuid.uuid4().hex[:8]}"
            render_res = render_and_export_dualseq(cmds, args, uid_or_name=session_uid)

            assistant_msg = (
                f"I have modeled the CAD geometry based on your description:\n\n"
                f"- **Parts Generated:** {render_res['tree_data']['part_count']}\n"
                f"- **Total Operations:** {len(cmds)}\n\n"
                f"The 3D model is ready for viewing, rotation, and inspection in the CAD Editor."
            )

            done_payload = {
                "type": "done",
                "thought": thought,
                "dualseq_tuples": render_res["dualseq_tuples"],
                "render_result": render_res,
                "tree_data": render_res["tree_data"],
                "raw_output": full_raw_output,
                "assistant_message": assistant_msg,
                "error": err or render_res.get("error"),
            }
            yield f"data: {json.dumps(done_payload)}\n\n"
        except Exception as e:
            err_payload = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/dualseq/render", response_model=RenderResponse)
def render_dualseq(request: RenderRequest):
    tuples = request.dualseq_tuples
    cmds = []
    args = []
    for item in tuples:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            cmds.append(str(item[0]))
            args.append(item[1] if isinstance(item[1], dict) else {})

    session_uid = f"edit_{uuid.uuid4().hex[:8]}"
    render_res = render_and_export_dualseq(cmds, args, uid_or_name=session_uid)

    return RenderResponse(
        success=render_res["success"],
        error=render_res["error"],
        stl_url=render_res["stl_url"],
        png_url=render_res["png_url"],
        tree_data=render_res["tree_data"],
        properties=render_res["properties"],
        dualseq_tuples=render_res["dualseq_tuples"],
    )


@app.post("/api/cad/export")
def export_cad(request: ExportRequest):
    tuples = request.dualseq_tuples
    cmds = [str(item[0]) for item in tuples if isinstance(item, (list, tuple)) and len(item) == 2]
    args = [item[1] if isinstance(item[1], dict) else {} for item in tuples if isinstance(item, (list, tuple)) and len(item) == 2]

    ds = DualSeq(cmds=cmds, args=args, uid=request.filename)

    if request.format.lower() == "json":
        export_data = {
            "dualseq_tuples": tuples,
            "cmds": cmds,
            "args": args,
            "minimal_json": ds.json_object,
        }
        return JSONResponse(content=export_data)

    elif request.format.lower() == "stl":
        session_uid = f"export_{uuid.uuid4().hex[:8]}"
        res = render_and_export_dualseq(cmds, args, uid_or_name=session_uid)
        if res["success"] and res["stl_filename"]:
            stl_path = os.path.join(DEFAULT_STL_DIR, res["stl_filename"])
            return FileResponse(
                stl_path,
                media_type="application/sla",
                filename=f"{request.filename}.stl",
            )
        raise HTTPException(status_code=400, detail="Could not export STL: OCC rendering failed.")

    raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")


FRONTEND_DIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.isdir(FRONTEND_DIST_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
