"""utils/render/render_img.py: Render OCC shape to isometric PNG using OCC display pipeline."""
import os
from OCC.Display.OCCViewer import Viewer3d
from OCC.Core.gp import gp_Dir
from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_WHITE, Quantity_TOC_RGB
from OCC.Core.V3d import V3d_DirectionalLight
from OCC.Extend.TopologyUtils import TopologyExplorer

def render_to_image(
    shape,
    filepath: str,
    size=(512, 512),
    face_color_rgb=(0.2, 0.2, 0.2),
    edge_color_rgb=(0, 0, 0),
    show_face_boundary=True,
    margin=0.1
) -> None:
    """
    Render a TopoDS_Shape to a high-fidelity isometric PNG using Python-OCC's
    internal OpenGL rendering pipeline via Viewer3d.

    Parameters:
    -----------
    shape : TopoDS_Shape
        The OpenCASCADE shape to render.
    filepath : str
        The output path for the saved PNG image.
    size : tuple of int, optional
        The width and height of the rendered image (default is (512, 512)).
    face_color_rgb : tuple of float, optional
        The RGB values for faces, normalized between 0 and 1.
    edge_color_rgb : tuple of float, optional
        The RGB values for edges, normalized between 0 and 1.
    show_face_boundary : bool, optional
        Whether to draw face boundaries (default is True).
    margin : float, optional
        The margin coefficient for view borders (default is 0.1 for 10% padding).
    """
    width, height = size
    viewer = Viewer3d()
    viewer.Create(phong_shading=True, create_default_lights=True)
    viewer.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    viewer.SetModeShaded()
    viewer.hide_triedron()
    viewer.EnableAntiAliasing()
    
    dir_light = V3d_DirectionalLight(gp_Dir(0, 0.5, -1), Quantity_Color(Quantity_NOC_WHITE))
    dir_light.SetEnabled(True)
    dir_light.SetIntensity(500.0)
    viewer.Viewer.AddLight(dir_light)
    viewer.Viewer.SetLightOn()

    viewer.default_drawer.EnableDrawHiddenLine()
    viewer.default_drawer.SetFaceBoundaryDraw(show_face_boundary)
    ais_context = viewer.GetContext()
    
    # Increase tessellation/deviation resolution for smoother rendering
    dc = ais_context.DeviationCoefficient()
    da = ais_context.DeviationAngle()
    factor = 10
    ais_context.SetDeviationCoefficient(dc / factor)
    ais_context.SetDeviationAngle(da / factor)
    
    topexp = TopologyExplorer(shape)
    for face in topexp.faces():
        if face is not None:
            viewer.DisplayShape(face, color=Quantity_Color(*face_color_rgb, Quantity_TOC_RGB))
    for edge in topexp.edges():
        if edge is not None:
            viewer.DisplayShape(edge, color=Quantity_Color(*edge_color_rgb, Quantity_TOC_RGB))
            
    viewer.SetSize(width, height)
    viewer.View.FitAll(margin)
    viewer.View.ZFitAll()
    
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    print("dasdasd")
    viewer.View.Dump(str(filepath))
