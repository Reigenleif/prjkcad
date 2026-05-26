from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Display.OCCViewer import Viewer3d

import os
import sys

class SuppressOutput:
    def __enter__(self):
        self.null_fds = os.open(os.devnull, os.O_RDWR)
        self.stdout_fd = os.dup(1)
        self.stderr_fd = os.dup(2)

        os.dup2(self.null_fds, 1)
        os.dup2(self.null_fds, 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.stdout_fd, 1)
        os.dup2(self.stderr_fd, 2)

        os.close(self.null_fds)
        os.close(self.stdout_fd)
        os.close(self.stderr_fd)


class Exporter:
    """
    Utility class for exporting OCC shapes to STL, STEP, and PNG formats.
    """
    def __init__(self) :
        self.stl_writer = StlAPI_Writer()
        self.step_writer = STEPControl_Writer()
        self.viewer = Viewer3d()
    
    def save_stl(self, shape, path):
        with SuppressOutput():
            self.stl_writer.Write(shape, str(path))


    def save_step(self, shape, path):
        with SuppressOutput():
            self.step_writer.Transfer(shape, STEPControl_AsIs)
            self.step_writer.Write(str(path))


    def save_png(self, shape, path):
        with SuppressOutput():
            self.viewer.Create()
            self.viewer.DisplayShape(shape)
            self.viewer.FitAll()
            self.viewer.View.Dump(str(path))