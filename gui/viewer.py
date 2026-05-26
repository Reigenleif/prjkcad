from OCC.Display.WebGl import threejs_renderer
from OCC.Core.TopoDS import TopoDS_Shape

def view(body: TopoDS_Shape):
    renderer = threejs_renderer.ThreejsRenderer()

    # display the shape
    renderer.DisplayShape(body)

    # render in browser / notebook
    renderer.render()