import json
import os
from pathlib import Path
from tqdm import tqdm

from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Display.OCCViewer import Viewer3d

from render.str_to_occ import CADSequenceToOCC
from render.export import Exporter

# from your_module import CADSequenceToOCC

JSON_FILES = [
    "data/sl_data/train.json",
    "data/sl_data/val.json",
    "data/sl_data/test.json"
]


# ----------------------------
# Tester
# ----------------------------
def test_converter(
    save_outputs=False,
    out_dir="debug_outputs",
    log_file="res.log"
):
    os.makedirs(out_dir, exist_ok=True)
    exporter = Exporter()

    # clear previous log
    with open(log_file, "w") as f:
        pass

    total = 0
    success = 0
    failed = 0

    for json_path in JSON_FILES:
        with open(json_path, "r") as f:
            data = json.load(f)

        for item in tqdm(data):
            total += 1
            print(f"Processing ID: {item.get('serial_num', total)}")  # 🔵 log the current ID being processed

            seq = item["command_sequence"]
            sid = item.get("serial_num", total)

            try:
                parser = CADSequenceToOCC(seq)
                shapes = parser.parse()

                if not shapes or not isinstance(shapes[0], TopoDS_Shape):
                    raise ValueError("Invalid shape output")

                success += 1

                if save_outputs:
                    base = Path(out_dir) / f"{sid}"
                    shape = shapes[0]

                    exporter.save_stl(shape, base.with_suffix(".stl"))
                    exporter.save_step(shape, base.with_suffix(".step"))

                    try:
                        exporter.save_png(shape, base.with_suffix(".png"))
                    except Exception:
                        pass  # ignore headless rendering errors

            except Exception:
                failed += 1

                # 🔴 log only the failed ID
                with open(log_file, "a") as f:
                    f.write(f"{sid}\n")

    # summary (only stdout output)
    print("====== RESULT ======")
    print(f"Total   : {total}")
    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    print(f"Rate    : {success/total:.2%}")
    print("====================")
    
if __name__ == "__main__":
    test_converter(
        save_outputs=True,
        out_dir="debug_outputs",
        log_file="res.log"
    )