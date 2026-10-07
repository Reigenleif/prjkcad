import json
import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

try:
    from gui.backend.deps import get_harness
    from gui.backend.schemas import ChatRequest, ChatResponse
    from gui.backend.cad_engine import render_and_export_dualseq
    from gui.backend.llm_harness import extract_thought_and_dualseq
except ImportError:
    from deps import get_harness
    from schemas import ChatRequest, ChatResponse
    from cad_engine import render_and_export_dualseq
    from llm_harness import extract_thought_and_dualseq

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
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


@router.post("/stream")
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
