import os
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from utils.representations.dual_seq.dual_seq import DualSeq
from gui.backend.schemas import RenderRequest, RenderResponse, ExportRequest
from gui.backend.cad_engine import render_and_export_dualseq, DEFAULT_STL_DIR

router = APIRouter(prefix="/api", tags=["cad"])


@router.post("/dualseq/render", response_model=RenderResponse)
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


@router.post("/cad/export")
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
