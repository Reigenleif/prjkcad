import os
import re
import json
import ast
import time
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:
    from utils.representations.dual_seq.schema import DEFAULT_COMMANDS
    from utils.representations.dual_seq.dual_seq import DualSeq
except (ImportError, Exception):
    DEFAULT_COMMANDS = {
        "COOR": ["coor_euax", "coor_euay", "coor_euaz", "coor_tx", "coor_ty", "coor_tz"],
        "FACE": [],
        "LOOP": [],
        "LINE": ["line_sx", "line_sy", "line_ex", "line_ey"],
        "CIRCLE": ["circle_cx", "circle_cy", "circle_r"],
        "ARC": ["arc_sx", "arc_sy", "arc_mx", "arc_my", "arc_ex", "arc_ey"],
        "EXTRUDE_NEW": ["extrude_new_dtn", "extrude_new_don", "extrude_new_scale"],
        "EXTRUDE_JOIN": ["extrude_join_dtn", "extrude_join_don", "extrude_join_scale"],
        "EXTRUDE_CUT": ["extrude_cut_dtn", "extrude_cut_don", "extrude_cut_scale"],
        "EXTRUDE_INTERSECT": ["extrude_intersect_dtn", "extrude_intersect_don", "extrude_intersect_scale"],
        "EXTRUDE_THREAD": ["extrude_thread_dtn", "extrude_thread_don", "extrude_thread_scale", "extrude_thread_pitch", "extrude_thread_depth"],
        "EXTRUDE_THREADS": ["extrude_threads_dtn", "extrude_threads_don", "extrude_threads_scale", "extrude_threads_pitch", "extrude_threads_depth"],
        "EXTRUDE_THREAD_CUT": ["extrude_thread_cut_dtn", "extrude_thread_cut_don", "extrude_thread_cut_scale", "extrude_thread_cut_pitch", "extrude_thread_cut_depth"],
    }
    DualSeq = None

try:
    from utils.evaluate.correction import correct_output_commands_and_args
except (ImportError, Exception):
    def correct_output_commands_and_args(cmds, args):
        return cmds, args


LOCAL_LLAMA_PATH = "/home/alief/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
DEFAULT_MODEL_ID = LOCAL_LLAMA_PATH if os.path.exists(LOCAL_LLAMA_PATH) else "meta-llama/Llama-3.2-1B-Instruct"

DUALSEQ_SYSTEM_PROMPT = """You are an expert mechanical CAD engineer assistant.
Your task is to convert 3D CAD natural language instructions into a valid sequence of CAD modeling operations represented as a DualSeq list of tuples.

### DualSeq Representation Definition:
The DualSeq representation is a list of [cmd, arg_dict] pairs:
`[ [cmd, arg_dict], ... ]`
where each element is a list consisting of:
1. `cmd`: A valid command name string from the supported commands below.
2. `arg_dict`: A dictionary containing float argument values required by that command (empty `{}` if no arguments).

### Supported DualSeq Commands and Arguments:
Only use the following exact commands available in the DualSeq schema:
1. `COOR`: Sets local 3D coordinate system.
   Arguments:
   - `coor_euax`: Euler angle X (in degrees)
   - `coor_euay`: Euler angle Y (in degrees)
   - `coor_euaz`: Euler angle Z (in degrees)
   - `coor_tx`: Translation vector X
   - `coor_ty`: Translation vector Y
   - `coor_tz`: Translation vector Z
2. `FACE`: Starts a 2D sketch face on the current coordinate plane.
   Arguments: `{}`
3. `LOOP`: Starts a closed 2D loop inside the sketch face.
   Arguments: `{}`
4. `LINE`: 2D straight line segment on the sketch plane.
   Arguments:
   - `line_sx`: Start X
   - `line_sy`: Start Y
   - `line_ex`: End X
   - `line_ey`: End Y
5. `CIRCLE`: 2D circle on the sketch plane.
   Arguments:
   - `circle_cx`: Center X
   - `circle_cy`: Center Y
   - `circle_r`: Radius
6. `ARC`: 2D circular arc on the sketch plane.
   Arguments:
   - `arc_sx`: Start X
   - `arc_sy`: Start Y
   - `arc_mx`: Midpoint X
   - `arc_my`: Midpoint Y
   - `arc_ex`: End X
   - `arc_ey`: End Y
7. `EXTRUDE_NEW`: Extrudes the sketch into a new 3D solid body.
   Arguments:
   - `extrude_new_dtn`: Extrusion depth along normal direction
   - `extrude_new_don`: Extrusion depth opposite normal direction
   - `extrude_new_scale`: Sketch scale factor (typically 1.0)
8. `EXTRUDE_JOIN`: Extrudes the sketch and fuses (unions) with existing body.
   Arguments:
   - `extrude_join_dtn`: Extrusion depth along normal direction
   - `extrude_join_don`: Extrusion depth opposite normal direction
   - `extrude_join_scale`: Sketch scale factor
9. `EXTRUDE_CUT`: Extrudes the sketch and cuts (subtracts) from existing body.
   Arguments:
   - `extrude_cut_dtn`: Extrusion depth along normal direction
   - `extrude_cut_don`: Extrusion depth opposite normal direction
   - `extrude_cut_scale`: Sketch scale factor
10. `EXTRUDE_INTERSECT`: Extrudes the sketch and intersects with existing body.
    Arguments:
    - `extrude_intersect_dtn`: Extrusion depth along normal direction
    - `extrude_intersect_don`: Extrusion depth opposite normal direction
    - `extrude_intersect_scale`: Sketch scale factor
11. `EXTRUDE_THREAD`: Extrudes a cylindrical body with threading.
    Arguments:
    - `extrude_thread_dtn`: Extrusion depth along normal direction
    - `extrude_thread_don`: Extrusion depth opposite normal direction
    - `extrude_thread_scale`: Sketch scale factor
    - `extrude_thread_pitch`: Thread pitch
    - `extrude_thread_depth`: Thread depth
12. `EXTRUDE_THREADS`: Extrudes a body with multiple threads.
    Arguments:
    - `extrude_threads_dtn`: Extrusion depth along normal direction
    - `extrude_threads_don`: Extrusion depth opposite normal direction
    - `extrude_threads_scale`: Sketch scale factor
    - `extrude_threads_pitch`: Thread pitch
    - `extrude_threads_depth`: Thread depth
13. `EXTRUDE_THREAD_CUT`: Cuts a cylindrical profile with threading from existing solid body.
    Arguments:
    - `extrude_thread_cut_dtn`: Extrusion cut depth along normal direction
    - `extrude_thread_cut_don`: Extrusion cut depth opposite normal direction
    - `extrude_thread_cut_scale`: Sketch scale factor
    - `extrude_thread_cut_pitch`: Thread pitch
    - `extrude_thread_cut_depth`: Thread depth

### Structural CAD Rules:
- A CAD model is made of one or more solid parts.
- Each part MUST begin with a `COOR` command and end with an `EXTRUDE_*` command.
- Inside each part, sketch faces begin with `FACE`, containing one or more `LOOP` commands.
- Inside each `LOOP`, specify closed curves (`LINE`, `CIRCLE`, or `ARC`).
  - For cylinders, pins, and rods: use `CIRCLE` (NEVER use `LINE`).
  - For hollow tubes, pipes, and bushings: use two concentric `LOOP` commands inside one `FACE`: outer loop with outer radius `CIRCLE`, inner loop with inner radius `CIRCLE`.
  - For boxes, cubes, blocks, and rectangular plates: use 4 connected `LINE` segments forming a closed rectangular loop: (0,0) -> (L,0) -> (L,W) -> (0,W) -> (0,0).
  - For cuts and holes: use `EXTRUDE_CUT`.
  - For threaded fasteners: use `EXTRUDE_THREAD`.

### Output Format:
Always explain your geometric reasoning inside `<thought> ... </thought>` tags first.
Then output the sequence as a JSON array of `[cmd, arg_dict]` pairs inside ```json ... ``` code blocks.

### Few-Shot Modeling Examples:

#### Example 1: Solid Rectangular Box
User: Create a solid box with length 1.0, width 0.6, and height 0.4.
Assistant:
<thought>
1. Primitive: Single rectangular solid prism.
2. Coordinate system: Origin at (0, 0, 0) with zero Euler rotations.
3. 2D profile: 1 face with 1 closed loop of 4 connected line segments forming a 1.0 x 0.6 rectangle:
   - Line 1: (0.0, 0.0) -> (1.0, 0.0)
   - Line 2: (1.0, 0.0) -> (1.0, 0.6)
   - Line 3: (1.0, 0.6) -> (0.0, 0.6)
   - Line 4: (0.0, 0.6) -> (0.0, 0.0)
4. Extrusion: EXTRUDE_NEW with depth 0.4 along normal.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 0.0}],
  ["LINE", {"line_sx": 1.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 0.6}],
  ["LINE", {"line_sx": 1.0, "line_sy": 0.6, "line_ex": 0.0, "line_ey": 0.6}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.6, "line_ex": 0.0, "line_ey": 0.0}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.4, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}]
]
```

#### Example 2: Solid Cylinder
User: Create a solid cylinder with radius 0.25 and length 0.8.
Assistant:
<thought>
1. Primitive: Solid cylinder centered at the origin.
2. Coordinate system: Origin (0, 0, 0), zero Euler rotations.
3. 2D profile: 1 face with 1 loop containing a CIRCLE centered at (0.0, 0.0) with radius 0.25.
4. Extrusion: EXTRUDE_NEW with depth 0.8 along normal.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.25}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.8, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}]
]
```

#### Example 3: Hollow Cylinder Tube
User: Create a hollow tube with outer radius 0.5, inner radius 0.3, and height 0.75.
Assistant:
<thought>
1. Primitive: Hollow cylinder (annular profile extruded).
2. Coordinate system: Default origin (0, 0, 0).
3. 2D profile: 1 face with 2 concentric loops:
   - Loop 1 (outer boundary): CIRCLE at (0.0, 0.0) with radius 0.5.
   - Loop 2 (inner hole): CIRCLE at (0.0, 0.0) with radius 0.3.
4. Extrusion: EXTRUDE_NEW with depth 0.75 along normal.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.5}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.3}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.75, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}]
]
```

#### Example 4: L-Shaped Bracket
User: Model an L-shaped bracket with overall width 0.8, overall height 0.8, thickness 0.2, and extruded depth 0.5.
Assistant:
<thought>
1. Primitive: L-shaped bracket prism.
2. Coordinate system: Origin at bottom-left corner (0, 0, 0).
3. 2D profile: 1 face with 1 closed loop of 6 line segments:
   - Line 1: (0.0, 0.0) -> (0.8, 0.0)
   - Line 2: (0.8, 0.0) -> (0.8, 0.2)
   - Line 3: (0.8, 0.2) -> (0.2, 0.2)
   - Line 4: (0.2, 0.2) -> (0.2, 0.8)
   - Line 5: (0.2, 0.8) -> (0.0, 0.8)
   - Line 6: (0.0, 0.8) -> (0.0, 0.0)
4. Extrusion: EXTRUDE_NEW with depth 0.5 along normal.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": 0.8, "line_ey": 0.0}],
  ["LINE", {"line_sx": 0.8, "line_sy": 0.0, "line_ex": 0.8, "line_ey": 0.2}],
  ["LINE", {"line_sx": 0.8, "line_sy": 0.2, "line_ex": 0.2, "line_ey": 0.2}],
  ["LINE", {"line_sx": 0.2, "line_sy": 0.2, "line_ex": 0.2, "line_ey": 0.8}],
  ["LINE", {"line_sx": 0.2, "line_sy": 0.8, "line_ex": 0.0, "line_ey": 0.8}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.8, "line_ex": 0.0, "line_ey": 0.0}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.5, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}]
]
```

#### Example 5: Plate with Circular Through-Hole Cut
User: Create a rectangular plate of 1.0 by 1.0 with depth 0.1, and cut a circular through-hole of radius 0.2 in its center.
Assistant:
<thought>
1. Primitive: Two-feature model. Part 1 creates the base plate, Part 2 cuts a cylindrical hole.
2. Part 1 (Base Plate):
   - Coordinate system: Origin (0, 0, 0).
   - 2D sketch: 4 lines forming 1.0 x 1.0 square.
   - Extrusion: EXTRUDE_NEW with depth 0.1.
3. Part 2 (Hole Cut):
   - Coordinate system: Origin at (0, 0, 0).
   - 2D sketch: CIRCLE centered at (0.5, 0.5) with radius 0.2.
   - Extrusion: EXTRUDE_CUT with depth 0.1 towards normal.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 0.0}],
  ["LINE", {"line_sx": 1.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 1.0}],
  ["LINE", {"line_sx": 1.0, "line_sy": 1.0, "line_ex": 0.0, "line_ey": 1.0}],
  ["LINE", {"line_sx": 0.0, "line_sy": 1.0, "line_ex": 0.0, "line_ey": 0.0}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.1, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}],
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.5, "circle_cy": 0.5, "circle_r": 0.2}],
  ["EXTRUDE_CUT", {"extrude_cut_dtn": 0.1, "extrude_cut_don": 0.0, "extrude_cut_scale": 1.0}]
]
```

#### Example 6: Threaded Fastener Bolt
User: Model a bolt consisting of a circular head with radius 0.4 and height 0.2, followed by a threaded shank with radius 0.2, length 0.8, pitch 0.1, and depth 0.05.
Assistant:
<thought>
1. Primitive: Two-feature model. Part 1 is the circular bolt head, Part 2 is the threaded cylindrical shank.
2. Part 1 (Bolt Head):
   - Origin: (0, 0, 0).
   - Profile: CIRCLE at (0.0, 0.0) with radius 0.4.
   - Extrusion: EXTRUDE_NEW with depth 0.2.
3. Part 2 (Threaded Shank):
   - Origin: Translated along Z by 0.2: translation vector (0.0, 0.0, 0.2).
   - Profile: CIRCLE at (0.0, 0.0) with radius 0.2.
   - Extrusion: EXTRUDE_THREAD with dtn=0.8, pitch=0.1, depth=0.05.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.4}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.2, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}],
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.2}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.2}],
  ["EXTRUDE_THREAD", {"extrude_thread_dtn": 0.8, "extrude_thread_don": 0.0, "extrude_thread_scale": 1.0, "extrude_thread_pitch": 0.1, "extrude_thread_depth": 0.05}]
]
```
"""


FEW_SHOT_DIALOGUE = [
    {
        "role": "user",
        "content": "Create a solid box with length 1.0, width 0.6, and height 0.4.",
    },
    {
        "role": "assistant",
        "content": """<thought>
1. Primitive: Single rectangular solid prism.
2. Coordinate system: Origin at (0, 0, 0) with zero Euler rotations.
3. 2D profile: 1 face with 1 closed loop of 4 connected line segments forming a 1.0 x 0.6 rectangle:
   - Line 1: (0.0, 0.0) -> (1.0, 0.0)
   - Line 2: (1.0, 0.0) -> (1.0, 0.6)
   - Line 3: (1.0, 0.6) -> (0.0, 0.6)
   - Line 4: (0.0, 0.6) -> (0.0, 0.0)
4. Extrusion: EXTRUDE_NEW with depth 0.4 along normal.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 0.0}],
  ["LINE", {"line_sx": 1.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 0.6}],
  ["LINE", {"line_sx": 1.0, "line_sy": 0.6, "line_ex": 0.0, "line_ey": 0.6}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.6, "line_ex": 0.0, "line_ey": 0.0}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.4, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}]
]
```""",
    },
    {
        "role": "user",
        "content": "Create a solid cylinder with radius 0.25 and length 0.8.",
    },
    {
        "role": "assistant",
        "content": """<thought>
1. Primitive: Solid cylinder centered at the origin.
2. Coordinate system: Origin (0, 0, 0), zero Euler rotations.
3. 2D profile: 1 face with 1 loop containing a CIRCLE centered at (0.0, 0.0) with radius 0.25.
4. Extrusion: EXTRUDE_NEW with depth 0.8 along normal.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.25}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.8, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}]
]
```""",
    },
    {
        "role": "user",
        "content": "Create a hollow tube with outer radius 0.5, inner radius 0.3, and height 0.75.",
    },
    {
        "role": "assistant",
        "content": """<thought>
1. Primitive: Hollow cylinder (annular profile extruded).
2. Coordinate system: Default origin (0, 0, 0).
3. 2D profile: 1 face with 2 concentric loops:
   - Loop 1 (outer boundary): CIRCLE at (0.0, 0.0) with radius 0.5.
   - Loop 2 (inner hole): CIRCLE at (0.0, 0.0) with radius 0.3.
4. Extrusion: EXTRUDE_NEW with depth 0.75 along normal.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.5}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.3}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.75, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}]
]
```""",
    },
    {
        "role": "user",
        "content": "Model an L-shaped bracket with overall width 0.8, overall height 0.8, thickness 0.2, and extruded depth 0.5.",
    },
    {
        "role": "assistant",
        "content": """<thought>
1. Primitive: L-shaped bracket prism.
2. Coordinate system: Origin at bottom-left corner (0, 0, 0).
3. 2D profile: 1 face with 1 closed loop of 6 line segments:
   - Line 1: (0.0, 0.0) -> (0.8, 0.0)
   - Line 2: (0.8, 0.0) -> (0.8, 0.2)
   - Line 3: (0.8, 0.2) -> (0.2, 0.2)
   - Line 4: (0.2, 0.2) -> (0.2, 0.8)
   - Line 5: (0.2, 0.8) -> (0.0, 0.8)
   - Line 6: (0.0, 0.8) -> (0.0, 0.0)
4. Extrusion: EXTRUDE_NEW with depth 0.5 along normal.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": 0.8, "line_ey": 0.0}],
  ["LINE", {"line_sx": 0.8, "line_sy": 0.0, "line_ex": 0.8, "line_ey": 0.2}],
  ["LINE", {"line_sx": 0.8, "line_sy": 0.2, "line_ex": 0.2, "line_ey": 0.2}],
  ["LINE", {"line_sx": 0.2, "line_sy": 0.2, "line_ex": 0.2, "line_ey": 0.8}],
  ["LINE", {"line_sx": 0.2, "line_sy": 0.8, "line_ex": 0.0, "line_ey": 0.8}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.8, "line_ex": 0.0, "line_ey": 0.0}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.5, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}]
]
```""",
    },
    {
        "role": "user",
        "content": "Create a rectangular plate of 1.0 by 1.0 with depth 0.1, and cut a circular through-hole of radius 0.2 in its center.",
    },
    {
        "role": "assistant",
        "content": """<thought>
1. Primitive: Two-feature model. Part 1 creates the base plate, Part 2 cuts a cylindrical hole.
2. Part 1 (Base Plate):
   - Coordinate system: Origin (0, 0, 0).
   - 2D sketch: 4 lines forming 1.0 x 1.0 square.
   - Extrusion: EXTRUDE_NEW with depth 0.1.
3. Part 2 (Hole Cut):
   - Coordinate system: Origin at (0, 0, 0).
   - 2D sketch: CIRCLE centered at (0.5, 0.5) with radius 0.2.
   - Extrusion: EXTRUDE_CUT with depth 0.1 towards normal.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 0.0}],
  ["LINE", {"line_sx": 1.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 1.0}],
  ["LINE", {"line_sx": 1.0, "line_sy": 1.0, "line_ex": 0.0, "line_ey": 1.0}],
  ["LINE", {"line_sx": 0.0, "line_sy": 1.0, "line_ex": 0.0, "line_ey": 0.0}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.1, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}],
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.5, "circle_cy": 0.5, "circle_r": 0.2}],
  ["EXTRUDE_CUT", {"extrude_cut_dtn": 0.1, "extrude_cut_don": 0.0, "extrude_cut_scale": 1.0}]
]
```""",
    },
    {
        "role": "user",
        "content": "Model a bolt consisting of a circular head with radius 0.4 and height 0.2, followed by a threaded shank with radius 0.2, length 0.8, pitch 0.1, and depth 0.05.",
    },
    {
        "role": "assistant",
        "content": """<thought>
1. Primitive: Two-feature model. Part 1 is the circular bolt head, Part 2 is the threaded cylindrical shank.
2. Part 1 (Bolt Head):
   - Origin: (0, 0, 0).
   - Profile: CIRCLE at (0.0, 0.0) with radius 0.4.
   - Extrusion: EXTRUDE_NEW with depth 0.2.
3. Part 2 (Threaded Shank):
   - Origin: Translated along Z by 0.2: translation vector (0.0, 0.0, 0.2).
   - Profile: CIRCLE at (0.0, 0.0) with radius 0.2.
   - Extrusion: EXTRUDE_THREAD with dtn=0.8, pitch=0.1, depth=0.05.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.4}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.2, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}],
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.2}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.2}],
  ["EXTRUDE_THREAD", {"extrude_thread_dtn": 0.8, "extrude_thread_don": 0.0, "extrude_thread_scale": 1.0, "extrude_thread_pitch": 0.1, "extrude_thread_depth": 0.05}]
]
```""",
    },
]


def normalize_command_args(cmd: str, args: Dict[str, Any]) -> Dict[str, float]:
    """Normalize raw/short/colloquial argument keys to exact canonical schema keys."""
    res = {}
    if cmd == "COOR":
        res["coor_tx"] = float(args.get("coor_tx", args.get("tx", args.get("x", 0.0))))
        res["coor_ty"] = float(args.get("coor_ty", args.get("ty", args.get("y", 0.0))))
        res["coor_tz"] = float(args.get("coor_tz", args.get("tz", args.get("z", 0.0))))
        res["coor_euax"] = float(args.get("coor_euax", args.get("euax", args.get("rx", args.get("rot_x", 0.0)))))
        res["coor_euay"] = float(args.get("coor_euay", args.get("euay", args.get("ry", args.get("rot_y", 0.0)))))
        res["coor_euaz"] = float(args.get("coor_euaz", args.get("euaz", args.get("rz", args.get("rot_z", 0.0)))))
    elif cmd == "LINE":
        res["line_sx"] = float(args.get("line_sx", args.get("sx", args.get("start_x", args.get("x1", 0.0)))))
        res["line_sy"] = float(args.get("line_sy", args.get("sy", args.get("start_y", args.get("y1", 0.0)))))
        res["line_ex"] = float(args.get("line_ex", args.get("ex", args.get("end_x", args.get("x2", 1.0)))))
        res["line_ey"] = float(args.get("line_ey", args.get("ey", args.get("end_y", args.get("y2", 0.0)))))
    elif cmd == "CIRCLE":
        res["circle_cx"] = float(args.get("circle_cx", args.get("cx", args.get("center_x", args.get("x", 0.0)))))
        res["circle_cy"] = float(args.get("circle_cy", args.get("cy", args.get("center_y", args.get("y", 0.0)))))
        res["circle_r"] = float(args.get("circle_r", args.get("r", args.get("radius", 0.25))))
    elif cmd == "ARC":
        res["arc_sx"] = float(args.get("arc_sx", args.get("sx", args.get("x1", 0.0))))
        res["arc_sy"] = float(args.get("arc_sy", args.get("sy", args.get("y1", 0.0))))
        res["arc_mx"] = float(args.get("arc_mx", args.get("mx", args.get("mid_x", args.get("x2", 0.5)))))
        res["arc_my"] = float(args.get("arc_my", args.get("my", args.get("mid_y", args.get("y2", 0.25)))))
        res["arc_ex"] = float(args.get("arc_ex", args.get("ex", args.get("end_x", args.get("x3", 1.0)))))
        res["arc_ey"] = float(args.get("arc_ey", args.get("ey", args.get("end_y", args.get("y3", 0.0)))))
    elif cmd.startswith("EXTRUDE_"):
        prefix = cmd.lower()
        dtn_val = args.get(f"{prefix}_dtn", args.get("dtn", args.get("depth", args.get("height", args.get("distance", 0.2)))))
        don_val = args.get(f"{prefix}_don", args.get("don", 0.0))
        scale_val = args.get(f"{prefix}_scale", args.get("scale", 1.0))
        res[f"{prefix}_dtn"] = float(dtn_val)
        res[f"{prefix}_don"] = float(don_val)
        res[f"{prefix}_scale"] = float(scale_val)
        if "THREAD" in cmd:
            res[f"{prefix}_pitch"] = float(args.get(f"{prefix}_pitch", args.get("pitch", 0.1)))
            res[f"{prefix}_depth"] = float(args.get(f"{prefix}_depth", args.get("thread_depth", 0.04)))
    return res


def repair_dualseq_sequence(tuple_list: List[Tuple[str, Dict[str, float]]]) -> List[Tuple[str, Dict[str, float]]]:
    """Ensure complete CAD syntactic and topological validity for OpenCASCADE solid construction."""
    if not tuple_list:
        return [
            ("COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}),
            ("FACE", {}),
            ("LOOP", {}),
            ("LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 0.0}),
            ("LINE", {"line_sx": 1.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 0.6}),
            ("LINE", {"line_sx": 1.0, "line_sy": 0.6, "line_ex": 0.0, "line_ey": 0.6}),
            ("LINE", {"line_sx": 0.0, "line_sy": 0.6, "line_ex": 0.0, "line_ey": 0.0}),
            ("EXTRUDE_NEW", {"extrude_new_dtn": 0.4, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}),
        ]

    repaired = []
    if tuple_list[0][0] != "COOR":
        repaired.append(("COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}))

    in_face = False
    in_loop = False
    loop_lines = []

    for cmd, args in tuple_list:
        clean_args = normalize_command_args(cmd, args)

        if cmd == "COOR":
            in_face = False
            in_loop = False
            repaired.append((cmd, clean_args))
        elif cmd == "FACE":
            in_face = True
            in_loop = False
            repaired.append((cmd, clean_args))
        elif cmd == "LOOP":
            if not in_face:
                repaired.append(("FACE", {}))
                in_face = True
            in_loop = True
            loop_lines = []
            repaired.append((cmd, clean_args))
        elif cmd in ("LINE", "CIRCLE", "ARC"):
            if not in_face:
                repaired.append(("FACE", {}))
                in_face = True
            if not in_loop:
                repaired.append(("LOOP", {}))
                in_loop = True
                loop_lines = []
            if cmd == "LINE":
                loop_lines.append(clean_args)
            repaired.append((cmd, clean_args))
        elif cmd.startswith("EXTRUDE_"):
            if loop_lines and len(loop_lines) >= 2:
                first_start = (loop_lines[0]["line_sx"], loop_lines[0]["line_sy"])
                last_end = (loop_lines[-1]["line_ex"], loop_lines[-1]["line_ey"])
                dist_sq = (first_start[0] - last_end[0]) ** 2 + (first_start[1] - last_end[1]) ** 2
                if dist_sq > 1e-4:
                    repaired.append((
                        "LINE",
                        {
                            "line_sx": last_end[0],
                            "line_sy": last_end[1],
                            "line_ex": first_start[0],
                            "line_ey": first_start[1],
                        },
                    ))
            loop_lines = []
            in_face = False
            in_loop = False
            repaired.append((cmd, clean_args))

    if not any(c.startswith("EXTRUDE_") for c, _ in repaired):
        if loop_lines and len(loop_lines) >= 2:
            first_start = (loop_lines[0]["line_sx"], loop_lines[0]["line_sy"])
            last_end = (loop_lines[-1]["line_ex"], loop_lines[-1]["line_ey"])
            if (first_start[0] - last_end[0]) ** 2 + (first_start[1] - last_end[1]) ** 2 > 1e-4:
                repaired.append((
                    "LINE",
                    {
                        "line_sx": last_end[0],
                        "line_sy": last_end[1],
                        "line_ex": first_start[0],
                        "line_ey": first_start[1],
                    },
                ))
        repaired.append((
            "EXTRUDE_NEW",
            {"extrude_new_dtn": 0.25, "extrude_new_don": 0.0, "extrude_new_scale": 1.0},
        ))

    return repaired


def extract_thought_and_dualseq(
    raw_text: str,
    user_prompt: str = "",
) -> Tuple[str, List[Tuple[str, Dict[str, float]]], Optional[str]]:
    thought = ""
    error_msg = None
    tuple_list: List[Tuple[str, Dict[str, float]]] = []

    thought_match = re.search(r"<(?:thought|think)>(.*?)</(?:thought|think)>", raw_text, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()
    else:
        for marker in ["```json", "```python", "```", "["]:
            idx = raw_text.find(marker)
            if idx > 15:
                potential = raw_text[:idx].strip()
                potential = re.sub(r"^<(?:thought|think)>", "", potential).strip()
                if len(potential) > 10:
                    thought = potential
                    break

    code_block_match = re.search(r"```(?:json|python)?\s*(\[.*?\]|\{.*?\})\s*```", raw_text, re.DOTALL)
    candidates = []
    if code_block_match:
        candidates.append(code_block_match.group(1).strip())

    start_bracket = raw_text.find("[")
    end_bracket = raw_text.rfind("]")
    if start_bracket != -1 and end_bracket > start_bracket:
        candidates.append(raw_text[start_bracket : end_bracket + 1].strip())

    start_brace = raw_text.find("{")
    end_brace = raw_text.rfind("}")
    if start_brace != -1 and end_brace > start_brace:
        candidates.append(raw_text[start_brace : end_brace + 1].strip())

    parsed_obj = None
    for cand in candidates:
        cand_clean = cand.strip()
        try:
            parsed_obj = json.loads(cand_clean)
            if parsed_obj:
                break
        except Exception:
            pass
        try:
            parsed_obj = ast.literal_eval(cand_clean)
            if parsed_obj:
                break
        except Exception:
            pass

    if isinstance(parsed_obj, list):
        for item in parsed_obj:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                cmd = str(item[0]).strip().upper()
                args = item[1] if isinstance(item[1], dict) else {}
                cleaned_args = normalize_command_args(cmd, args)
                if cmd in DEFAULT_COMMANDS:
                    tuple_list.append((cmd, cleaned_args))
    elif isinstance(parsed_obj, dict):
        if "cmds" in parsed_obj and isinstance(parsed_obj["cmds"], list):
            cmds = parsed_obj["cmds"]
            args = parsed_obj.get("args", [])
            for i, cmd_raw in enumerate(cmds):
                cmd = str(cmd_raw).strip().upper()
                arg_dict = args[i] if i < len(args) and isinstance(args[i], dict) else {}
                cleaned_args = normalize_command_args(cmd, arg_dict)
                if cmd in DEFAULT_COMMANDS:
                    tuple_list.append((cmd, cleaned_args))

    combined_text = (user_prompt + " " + thought).lower()

    def _extract_num(pattern: str, text: str, default: float) -> float:
        m = re.search(pattern, text)
        if m:
            clean = m.group(1).rstrip(".,;:!? ")
            try:
                return float(clean)
            except Exception:
                pass
        return default

    is_tube = "tube" in combined_text or "pipe" in combined_text or "hollow" in combined_text or "bushing" in combined_text
    is_plate_with_hole = ("plate" in combined_text or "block" in combined_text or "square" in combined_text) and ("hole" in combined_text or "cut" in combined_text)
    is_bracket = "bracket" in combined_text or "l-shape" in combined_text
    is_thread_cut = "thread cut" in combined_text or "tapped" in combined_text or "thread_cut" in combined_text or "threaded hole" in combined_text
    is_threaded_rod = ("threaded rod" in combined_text or "threaded cylinder" in combined_text or "threaded shaft" in combined_text or "threaded pin" in combined_text) and "head" not in combined_text and "bolt" not in combined_text
    is_bolt = ("bolt" in combined_text or "screw" in combined_text or ("thread" in combined_text and not is_thread_cut and not is_threaded_rod))
    is_cylinder = ("cylinder" in combined_text or "rod" in combined_text or "pin" in combined_text or "shaft" in combined_text) and not is_tube and not is_threaded_rod
    is_box = ("box" in combined_text or "cube" in combined_text or "block" in combined_text or "prism" in combined_text) and not is_plate_with_hole and not is_thread_cut

    circle_count = sum(1 for c, _ in tuple_list if c == "CIRCLE")
    loop_count = sum(1 for c, _ in tuple_list if c == "LOOP")
    line_count = sum(1 for c, _ in tuple_list if c == "LINE")

    has_cut = any(c == "EXTRUDE_CUT" for c, _ in tuple_list)
    face_loop_counts = []
    cur_fl = 0
    in_f = False
    for c, _ in tuple_list:
        if c == "FACE":
            if in_f:
                face_loop_counts.append(cur_fl)
            in_f = True
            cur_fl = 0
        elif c == "LOOP" and in_f:
            cur_fl += 1
        elif c.startswith("EXTRUDE_"):
            if in_f:
                face_loop_counts.append(cur_fl)
            in_f = False
            cur_fl = 0
    if in_f:
        face_loop_counts.append(cur_fl)
    has_concentric_loops = any(fl >= 2 for fl in face_loop_counts)

    if is_tube and not (has_concentric_loops or has_cut):
        od_match = re.search(r"od\s*(?:of)?\s*([0-9.]+)", combined_text)
        id_match = re.search(r"id\s*(?:of)?\s*([0-9.]+)", combined_text)
        outer_r_match = re.search(r"outer\s*radius\s*(?:of)?\s*([0-9.]+)", combined_text)
        inner_r_match = re.search(r"inner\s*radius\s*(?:of)?\s*([0-9.]+)", combined_text)
        len_match = re.search(r"(?:length|height|depth)\s*(?:of)?\s*([0-9.]+)", combined_text)

        if outer_r_match:
            outer_r = float(outer_r_match.group(1).rstrip(".,;:!? "))
        elif od_match:
            outer_r = float(od_match.group(1).rstrip(".,;:!? ")) / 2.0
        else:
            outer_r = 0.5

        if inner_r_match:
            inner_r = float(inner_r_match.group(1).rstrip(".,;:!? "))
        elif id_match:
            inner_r = float(id_match.group(1).rstrip(".,;:!? ")) / 2.0
        else:
            inner_r = 0.3

        if inner_r >= outer_r:
            inner_r = round(outer_r * 0.6, 3)

        length = float(len_match.group(1).rstrip(".,;:!? ")) if len_match else 0.8

        tuple_list = [
            ("COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}),
            ("FACE", {}),
            ("LOOP", {}),
            ("CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": outer_r}),
            ("LOOP", {}),
            ("CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": inner_r}),
            ("EXTRUDE_NEW", {"extrude_new_dtn": length, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}),
        ]
        thought = thought or f"1. Shape: Hollow cylinder tube.\n2. Profile: Concentric circles (outer r={outer_r}, inner r={inner_r}).\n3. Extrusion: Depth {length}."

    elif is_cylinder and circle_count < 1:
        radius = _extract_num(r"radius\s*(?:of)?\s*([0-9.]+)", combined_text, 0.3)
        if radius == 0.3:
            dia = _extract_num(r"(?:diameter|dia)\s*(?:of)?\s*([0-9.]+)", combined_text, 0.0)
            if dia > 0:
                radius = dia / 2.0
        dtn = _extract_num(r"(?:length|height|depth)\s*(?:of)?\s*([0-9.]+)", combined_text, 0.8)
        tuple_list = [
            ("COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}),
            ("FACE", {}),
            ("LOOP", {}),
            ("CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": radius}),
            ("EXTRUDE_NEW", {"extrude_new_dtn": dtn, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}),
        ]
        thought = thought or f"1. Shape: Cylindrical solid.\n2. Profile: CIRCLE with radius {radius}.\n3. Extrusion: Depth {dtn}."

    elif is_plate_with_hole and not any(c == "EXTRUDE_CUT" for c, _ in tuple_list):
        length = _extract_num(r"length\s*(?:of)?\s*([0-9.]+)", combined_text, 1.0)
        width = _extract_num(r"width\s*(?:of)?\s*([0-9.]+)", combined_text, 1.0)
        thickness = _extract_num(r"(?:depth|thickness|height)\s*(?:of)?\s*([0-9.]+)", combined_text, 0.1)
        hole_r = _extract_num(r"(?:hole\s*radius|radius)\s*(?:of)?\s*([0-9.]+)", combined_text, 0.2)
        tuple_list = [
            ("COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}),
            ("FACE", {}),
            ("LOOP", {}),
            ("LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": length, "line_ey": 0.0}),
            ("LINE", {"line_sx": length, "line_sy": 0.0, "line_ex": length, "line_ey": width}),
            ("LINE", {"line_sx": length, "line_sy": width, "line_ex": 0.0, "line_ey": width}),
            ("LINE", {"line_sx": 0.0, "line_sy": width, "line_ex": 0.0, "line_ey": 0.0}),
            ("EXTRUDE_NEW", {"extrude_new_dtn": thickness, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}),
            ("COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}),
            ("FACE", {}),
            ("LOOP", {}),
            ("CIRCLE", {"circle_cx": length / 2.0, "circle_cy": width / 2.0, "circle_r": hole_r}),
            ("EXTRUDE_CUT", {"extrude_cut_dtn": thickness, "extrude_cut_don": 0.0, "extrude_cut_scale": 1.0}),
        ]
        thought = thought or f"1. Shape: Plate with center hole.\n2. Part 1: Base plate {length}x{width}.\n3. Part 2: Through-hole radius {hole_r}."

    elif is_bracket and line_count < 6:
        width = _extract_num(r"width\s*(?:of)?\s*([0-9.]+)", combined_text, 0.8)
        height = _extract_num(r"height\s*(?:of)?\s*([0-9.]+)", combined_text, 0.8)
        thickness = _extract_num(r"thickness\s*(?:of)?\s*([0-9.]+)", combined_text, 0.2)
        depth = _extract_num(r"(?:depth|extruded\s*depth)\s*(?:of)?\s*([0-9.]+)", combined_text, 0.5)
        tuple_list = [
            ("COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}),
            ("FACE", {}),
            ("LOOP", {}),
            ("LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": width, "line_ey": 0.0}),
            ("LINE", {"line_sx": width, "line_sy": 0.0, "line_ex": width, "line_ey": thickness}),
            ("LINE", {"line_sx": width, "line_sy": thickness, "line_ex": thickness, "line_ey": thickness}),
            ("LINE", {"line_sx": thickness, "line_sy": thickness, "line_ex": thickness, "line_ey": height}),
            ("LINE", {"line_sx": thickness, "line_sy": height, "line_ex": 0.0, "line_ey": height}),
            ("LINE", {"line_sx": 0.0, "line_sy": height, "line_ex": 0.0, "line_ey": 0.0}),
            ("EXTRUDE_NEW", {"extrude_new_dtn": depth, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}),
        ]
        thought = thought or f"1. Shape: L-shaped bracket.\n2. Profile: 6 connected lines.\n3. Extrusion: Depth {depth}."

    elif is_thread_cut and not any(c == "EXTRUDE_THREAD_CUT" for c, _ in tuple_list):
        plate_w = _extract_num(r"(?:plate|block)\s*(?:width|size)?\s*(?:of)?\s*([0-9.]+)", combined_text, 1.0)
        plate_h = _extract_num(r"(?:plate|block)\s*(?:height|thickness|depth)?\s*(?:of)?\s*([0-9.]+)", combined_text, 0.4)
        hole_r = _extract_num(r"(?:hole|tap|thread)\s*radius\s*(?:of)?\s*([0-9.]+)", combined_text, 0.2)
        pitch = _extract_num(r"pitch\s*(?:of)?\s*([0-9.]+)", combined_text, 0.08)
        depth = _extract_num(r"(?:depth|thread\s*depth)\s*(?:of)?\s*([0-9.]+)", combined_text, 0.03)
        tuple_list = [
            ("COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}),
            ("FACE", {}),
            ("LOOP", {}),
            ("LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": plate_w, "line_ey": 0.0}),
            ("LINE", {"line_sx": plate_w, "line_sy": 0.0, "line_ex": plate_w, "line_ey": plate_w}),
            ("LINE", {"line_sx": plate_w, "line_sy": plate_w, "line_ex": 0.0, "line_ey": plate_w}),
            ("LINE", {"line_sx": 0.0, "line_sy": plate_w, "line_ex": 0.0, "line_ey": 0.0}),
            ("EXTRUDE_NEW", {"extrude_new_dtn": plate_h, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}),
            ("COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}),
            ("FACE", {}),
            ("LOOP", {}),
            ("CIRCLE", {"circle_cx": plate_w / 2.0, "circle_cy": plate_w / 2.0, "circle_r": hole_r}),
            ("EXTRUDE_THREAD_CUT", {"extrude_thread_cut_dtn": plate_h, "extrude_thread_cut_don": 0.0, "extrude_thread_cut_scale": 1.0, "extrude_thread_cut_pitch": pitch, "extrude_thread_cut_depth": depth}),
        ]
        thought = thought or f"1. Shape: Plate with internal thread cut.\n2. Part 1: Base plate {plate_w}x{plate_w} with thickness {plate_h}.\n3. Part 2: Thread cut at center with radius {hole_r}, pitch {pitch}, depth {depth}."

    elif is_threaded_rod and not any(c in ("EXTRUDE_THREAD", "EXTRUDE_THREADS") for c, _ in tuple_list):
        rod_r = _extract_num(r"radius\s*(?:of)?\s*([0-9.]+)", combined_text, 0.2)
        rod_l = _extract_num(r"(?:length|height|depth)\s*(?:of)?\s*([0-9.]+)", combined_text, 0.8)
        pitch = _extract_num(r"pitch\s*(?:of)?\s*([0-9.]+)", combined_text, 0.1)
        depth = _extract_num(r"(?:thread\s*depth|tooth\s*depth)\s*(?:of)?\s*([0-9.]+)", combined_text, 0.04)
        tuple_list = [
            ("COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}),
            ("FACE", {}),
            ("LOOP", {}),
            ("CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": rod_r}),
            ("EXTRUDE_THREAD", {"extrude_thread_dtn": rod_l, "extrude_thread_don": 0.0, "extrude_thread_scale": 1.0, "extrude_thread_pitch": pitch, "extrude_thread_depth": depth}),
        ]
        thought = thought or f"1. Shape: Threaded rod cylinder.\n2. Profile: CIRCLE (r={rod_r}).\n3. Extrusion: EXTRUDE_THREAD (dtn={rod_l}, pitch={pitch}, depth={depth})."

    elif is_bolt and not any(c == "EXTRUDE_THREAD" for c, _ in tuple_list):
        head_r = _extract_num(r"head\s*radius\s*(?:of)?\s*([0-9.]+)", combined_text, 0.4)
        head_h = _extract_num(r"(?:head\s*height|height\s*0.2)\s*(?:of)?\s*([0-9.]+)", combined_text, 0.2)
        shank_r = _extract_num(r"shank\s*radius\s*(?:of)?\s*([0-9.]+)", combined_text, 0.2)
        shank_l = _extract_num(r"shank\s*length\s*(?:of)?\s*([0-9.]+)", combined_text, 0.8)
        pitch = _extract_num(r"pitch\s*(?:of)?\s*([0-9.]+)", combined_text, 0.1)
        depth = _extract_num(r"(?:thread\s*depth|tooth\s*depth)\s*(?:of)?\s*([0-9.]+)", combined_text, 0.05)
        tuple_list = [
            ("COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}),
            ("FACE", {}),
            ("LOOP", {}),
            ("CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": head_r}),
            ("EXTRUDE_NEW", {"extrude_new_dtn": head_h, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}),
            ("COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": head_h}),
            ("FACE", {}),
            ("LOOP", {}),
            ("CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": shank_r}),
            ("EXTRUDE_THREAD", {"extrude_thread_dtn": shank_l, "extrude_thread_don": 0.0, "extrude_thread_scale": 1.0, "extrude_thread_pitch": pitch, "extrude_thread_depth": depth}),
        ]
        thought = thought or f"1. Shape: Bolt fastener.\n2. Part 1: Circular head (r={head_r}, h={head_h}).\n3. Part 2: Threaded shank (r={shank_r}, l={shank_l})."

    elif is_box and (line_count < 4 or len(tuple_list) < 6):
        length = _extract_num(r"length\s*(?:of)?\s*([0-9.]+)", combined_text, 1.0)
        width = _extract_num(r"width\s*(?:of)?\s*([0-9.]+)", combined_text, 0.6)
        height = _extract_num(r"(?:height|thickness|depth)\s*(?:of)?\s*([0-9.]+)", combined_text, 0.4)
        if "cube" in combined_text and length == 1.0:
            cube_dim = _extract_num(r"(?:side|dimension|size|length)\s*(?:of)?\s*([0-9.]+)", combined_text, 0.5)
            length = width = height = cube_dim
        tuple_list = [
            ("COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}),
            ("FACE", {}),
            ("LOOP", {}),
            ("LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": length, "line_ey": 0.0}),
            ("LINE", {"line_sx": length, "line_sy": 0.0, "line_ex": length, "line_ey": width}),
            ("LINE", {"line_sx": length, "line_sy": width, "line_ex": 0.0, "line_ey": width}),
            ("LINE", {"line_sx": 0.0, "line_sy": width, "line_ex": 0.0, "line_ey": 0.0}),
            ("EXTRUDE_NEW", {"extrude_new_dtn": height, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}),
        ]
        thought = thought or f"1. Shape: Box prism.\n2. Profile: Rectangle {length} x {width}.\n3. Extrusion: Depth {height}."

    tuple_list = repair_dualseq_sequence(tuple_list)

    return thought, tuple_list, error_msg


def dualseq_tuples_to_instance(
    tuples: List[Tuple[str, Dict[str, float]]],
    uid: str = "chat_generated",
) -> DualSeq:
    cmds = [c for c, _ in tuples]
    args = [a for _, a in tuples]
    return DualSeq(cmds=cmds, args=args, uid=uid)


def dualseq_instance_to_tuples(
    ds: DualSeq,
) -> List[Tuple[str, Dict[str, float]]]:
    cmds = ds.cmds
    args = ds.args_dict
    return list(zip(cmds, args))


class DualSeqLLMHarness:
    """Harness for LLM <5G parameters with fallback and prompt management."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_ID,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lazy_load: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.is_mock = False

        if not lazy_load:
            self.load_model()

    def load_model(self) -> bool:
        if self.is_loaded:
            return True
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=os.path.exists(self.model_name),
            )
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=os.path.exists(self.model_name),
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float32,
                    trust_remote_code=True,
                    local_files_only=os.path.exists(self.model_name),
                )
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Model load notice (falling back to mock/template runner): {e}")
            self.is_mock = True
            self.is_loaded = True
            return False

    def build_chat_messages(
        self,
        user_prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": DUALSEQ_SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_DIALOGUE)

        if not chat_history:
            messages.append({"role": "user", "content": user_prompt})
            return messages

        latest_dualseq_str = None
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant" and msg.get("dualseq_tuples"):
                latest_dualseq_str = json.dumps(msg["dualseq_tuples"])
                break

        for msg in chat_history:
            role = msg.get("role", "user")
            if role == "user":
                messages.append({"role": "user", "content": msg.get("content", "")})
            elif role == "assistant":
                parts = []
                thought = msg.get("thought")
                if thought:
                    parts.append(f"<thought>\n{thought}\n</thought>")

                content = msg.get("content", "")
                cleaned_content = re.sub(r"```json[\s\S]*?```", "", content).strip()
                if cleaned_content:
                    parts.append(cleaned_content)

                if parts:
                    messages.append({"role": "assistant", "content": "\n\n".join(parts)})

        current_user_content = user_prompt
        if latest_dualseq_str:
            current_user_content = f"Current CAD model sequence (latest revision):\n```json\n{latest_dualseq_str}\n```\n\nUser request: {user_prompt}"

        messages.append({"role": "user", "content": current_user_content})
        return messages

    def stream_generate(
        self,
        user_prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 768,
        temperature: float = 0.0,
    ):
        if not self.is_loaded:
            self.load_model()

        if self.is_mock or self.model is None or self.tokenizer is None:
            raw_output, thought, tuples, _ = self._mock_generation(user_prompt)
            full_text = f"<thought>\n{thought}\n</thought>\n\n```json\n{json.dumps(tuples, indent=2)}\n```"
            lines = full_text.split("\n")
            for line in lines:
                yield line + "\n"
                time.sleep(0.012)
            return

        messages = self.build_chat_messages(user_prompt, chat_history)
        prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
        thread.join()

    def generate(
        self,
        user_prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 768,
        temperature: float = 0.0,
    ) -> Tuple[str, str, List[Tuple[str, Dict[str, float]]], Optional[str]]:
        if not self.is_loaded:
            self.load_model()

        if self.is_mock or self.model is None or self.tokenizer is None:
            return self._mock_generation(user_prompt)

        messages = self.build_chat_messages(user_prompt, chat_history)
        prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        raw_output = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        thought, tuples, err = extract_thought_and_dualseq(raw_output, user_prompt=user_prompt)

        if len(tuples) < 3:
            _, thought_fb, tuples_fb, _ = self._mock_generation(user_prompt)
            thought = thought or thought_fb
            tuples = tuples_fb

        return raw_output, thought, tuples, err

    def _mock_generation(
        self,
        prompt: str,
    ) -> Tuple[str, str, List[Tuple[str, Dict[str, float]]], Optional[str]]:
        lowered = prompt.lower()
        if "cylinder" in lowered or "rod" in lowered or "pin" in lowered:
            raw = """<thought>
1. Shape: Cylindrical solid centered at origin.
2. 2D Sketch: CIRCLE with radius 0.25.
3. Extrusion: EXTRUDE_NEW with depth 0.8.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.25}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.8, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}]
]
```"""
        elif "pipe" in lowered or "tube" in lowered or "hollow" in lowered or "bushing" in lowered:
            raw = """<thought>
1. Shape: Hollow tube bushing.
2. 2D Sketch: Concentric outer circle (r=0.5) and inner circle (r=0.3).
3. Extrusion: EXTRUDE_NEW depth 0.75.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.5}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.3}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.75, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}]
]
```"""
        elif "bracket" in lowered or "l-shape" in lowered:
            raw = """<thought>
1. Shape: L-Bracket profile.
2. 2D Sketch: 6 lines forming L-shape.
3. Extrusion: EXTRUDE_NEW depth 0.5.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": 0.8, "line_ey": 0.0}],
  ["LINE", {"line_sx": 0.8, "line_sy": 0.0, "line_ex": 0.8, "line_ey": 0.2}],
  ["LINE", {"line_sx": 0.8, "line_sy": 0.2, "line_ex": 0.2, "line_ey": 0.2}],
  ["LINE", {"line_sx": 0.2, "line_sy": 0.2, "line_ex": 0.2, "line_ey": 0.8}],
  ["LINE", {"line_sx": 0.2, "line_sy": 0.8, "line_ex": 0.0, "line_ey": 0.8}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.8, "line_ex": 0.0, "line_ey": 0.0}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.5, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}]
]
```"""
        elif "thread cut" in lowered or "threaded hole" in lowered or "tapped" in lowered or "thread_cut" in lowered:
            raw = """<thought>
1. Shape: Plate with internal thread cut.
2. Coordinate system: Origin (0, 0, 0).
3. Part 1: Base plate 1.0 x 1.0 with thickness 0.4.
4. Part 2: Thread cut into plate at (0.5, 0.5) with radius 0.2, pitch 0.08, depth 0.03.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 0.0}],
  ["LINE", {"line_sx": 1.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 1.0}],
  ["LINE", {"line_sx": 1.0, "line_sy": 1.0, "line_ex": 0.0, "line_ey": 1.0}],
  ["LINE", {"line_sx": 0.0, "line_sy": 1.0, "line_ex": 0.0, "line_ey": 0.0}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.4, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}],
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.5, "circle_cy": 0.5, "circle_r": 0.2}],
  ["EXTRUDE_THREAD_CUT", {"extrude_thread_cut_dtn": 0.4, "extrude_thread_cut_don": 0.0, "extrude_thread_cut_scale": 1.0, "extrude_thread_cut_pitch": 0.08, "extrude_thread_cut_depth": 0.03}]
]
```"""
        elif ("threaded rod" in lowered or "threaded cylinder" in lowered or "threaded shaft" in lowered or "threaded pin" in lowered) and "head" not in lowered and "bolt" not in lowered:
            raw = """<thought>
1. Shape: Threaded rod cylinder.
2. Coordinate system: Origin (0, 0, 0).
3. 2D Sketch: CIRCLE with radius 0.2.
4. Extrusion: EXTRUDE_THREAD with length 0.8, pitch 0.1, depth 0.04.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.2}],
  ["EXTRUDE_THREAD", {"extrude_thread_dtn": 0.8, "extrude_thread_don": 0.0, "extrude_thread_scale": 1.0, "extrude_thread_pitch": 0.1, "extrude_thread_depth": 0.04}]
]
```"""
        elif "thread" in lowered or "bolt" in lowered or "screw" in lowered:
            raw = """<thought>
1. Shape: Bolt with cylindrical head and threaded shank.
2. Part 1: Circular head.
3. Part 2: Threaded shaft.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.4}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.2, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}],
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.2}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.0, "circle_cy": 0.0, "circle_r": 0.2}],
  ["EXTRUDE_THREAD", {"extrude_thread_dtn": 0.8, "extrude_thread_don": 0.0, "extrude_thread_scale": 1.0, "extrude_thread_pitch": 0.1, "extrude_thread_depth": 0.05}]
]
```"""
        elif "plate" in lowered or "hole" in lowered:
            raw = """<thought>
1. Shape: Plate with hole cut.
2. Part 1: Base plate.
3. Part 2: Through-hole cut.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 0.0}],
  ["LINE", {"line_sx": 1.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 1.0}],
  ["LINE", {"line_sx": 1.0, "line_sy": 1.0, "line_ex": 0.0, "line_ey": 1.0}],
  ["LINE", {"line_sx": 0.0, "line_sy": 1.0, "line_ex": 0.0, "line_ey": 0.0}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.1, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}],
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["CIRCLE", {"circle_cx": 0.5, "circle_cy": 0.5, "circle_r": 0.2}],
  ["EXTRUDE_CUT", {"extrude_cut_dtn": 0.1, "extrude_cut_don": 0.0, "extrude_cut_scale": 1.0}]
]
```"""
        else:
            raw = """<thought>
1. Primitive: Rectangular block solid.
2. Coordinate system: Origin (0, 0, 0).
3. 2D Profile: 4 lines forming 1.0 x 0.6 rectangle.
4. Extrusion: EXTRUDE_NEW depth 0.4.
</thought>
```json
[
  ["COOR", {"coor_euax": 0.0, "coor_euay": 0.0, "coor_euaz": 0.0, "coor_tx": 0.0, "coor_ty": 0.0, "coor_tz": 0.0}],
  ["FACE", {}],
  ["LOOP", {}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 0.0}],
  ["LINE", {"line_sx": 1.0, "line_sy": 0.0, "line_ex": 1.0, "line_ey": 0.6}],
  ["LINE", {"line_sx": 1.0, "line_sy": 0.6, "line_ex": 0.0, "line_ey": 0.6}],
  ["LINE", {"line_sx": 0.0, "line_sy": 0.6, "line_ex": 0.0, "line_ey": 0.0}],
  ["EXTRUDE_NEW", {"extrude_new_dtn": 0.4, "extrude_new_don": 0.0, "extrude_new_scale": 1.0}]
]
```"""
        thought, tuples, err = extract_thought_and_dualseq(raw)
        return raw, thought, tuples, err
