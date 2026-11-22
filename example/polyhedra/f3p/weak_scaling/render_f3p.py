# Render a snapshot of an F3P weak-scaling case (beam replicated along y).
# Usage: pvpython --force-offscreen-rendering render_f3p.py case_yN/F3P_WS_YN <step> <image.png> [color_by]
#   color_by: copy (default, fragments coloured by the copy they come from) | velocity
import sys, os
from paraview.simple import *

out_dir, step, png = sys.argv[1], int(sys.argv[2]), sys.argv[3]
color_by = sys.argv[4] if len(sys.argv) > 4 else "copy"
pv = os.path.join(out_dir, "ParaviewOutputFiles")

frag = XMLPartitionedUnstructuredGridReader(FileName=[os.path.join(pv, f"exadem_{step:010d}.pvtu")])
view = CreateView("RenderView")
view.ViewSize = [1600, 1000]
view.Background = [1, 1, 1]
try:
    view.UseColorPaletteForBackground = 0
except AttributeError:
    pass
view.OrientationAxesVisibility = 0

if color_by == "copy":
    calc = Calculator(Input=frag)
    calc.AttributeType = "Cell Data"
    calc.ResultArrayName = "copy"
    calc.Function = "floor(id/1000)"
    disp = Show(calc, view)
    ColorBy(disp, ("CELLS", "copy"))
    lut = GetColorTransferFunction("copy")
    lut.InterpretValuesAsCategories = 1
    lut.Annotations = ["0", "copie 1", "1", "copie 2", "2", "copie 3"]
    lut.IndexedColors = [0.40, 0.76, 0.65, 0.99, 0.55, 0.38, 0.55, 0.63, 0.80]
else:
    disp = Show(frag, view)
    ColorBy(disp, ("CELLS", "velocity", "Magnitude"))
    lut = GetColorTransferFunction("velocity")
    for preset in ("Viridis (matplotlib)", "Viridis", "Inferno (matplotlib)", "Cool to Warm"):
        try:
            lut.ApplyPreset(preset, True); break
        except RuntimeError:
            pass
    disp.RescaleTransferFunctionToDataRange(False, True)
    disp.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(lut, view)
    bar.Title = "|v|"
    bar.ComponentTitle = ""
    bar.TitleColor = bar.LabelColor = [0.1, 0.1, 0.1]
disp.EdgeColor = [0.15, 0.15, 0.15]
disp.SetRepresentationType("Surface With Edges")

# drivers: the exaDEM vtk files only hold the 2 end points of each cylinder (and are written
# at step 0 only), so the cylinders are drawn here (radius = 0.01 from the .shp files).
# Cylinder half-length read from punch.shp. Punch: LINEAR_MOTION along -z at 5 m/s, dt = 1e-6.
dt, v_punch, r_cyl = 1e-6, 5.0, 0.01
y_cyl = float(open(os.path.join(os.path.dirname(os.path.abspath(out_dir)), "punch.shp")).read().split("nv 2")[1].split()[4])
cylinders = {"support_left": (-0.9, -0.261), "support_right": (0.9, -0.261),
             "punch": (0.0, 0.261 - v_punch * dt * step)}
for name, (x, z) in cylinders.items():
    line = Line(Point1=[x, -y_cyl, z], Point2=[x, y_cyl, z], Resolution=1)
    tube = Tube(Input=line, Radius=r_cyl, NumberofSides=32, Capping=1)
    d = Show(tube, view)
    d.ColorArrayName = ["POINTS", ""]
    d.DiffuseColor = [0.85, 0.2, 0.2] if name == "punch" else [0.3, 0.3, 0.35]

Render(view)  # first render resets the camera: set it afterwards
cam = view.GetActiveCamera()
cam.SetPosition(2.6, -3.6, 2.2)
cam.SetFocalPoint(0, 0, -0.1)
cam.SetViewUp(0, 0, 1)
cam.SetViewAngle(30)
view.CenterOfRotation = [0, 0, 0]
Render(view)
SaveScreenshot(png, view, ImageResolution=[1600, 1000], TransparentBackground=0)
print("wrote", png)
