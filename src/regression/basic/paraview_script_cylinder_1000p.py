from paraview.simple import *
Cylinder()
SetProperties(Resolution=300)
SetProperties(Capping=False)
SetProperties(Radius=9)
SetProperties(Center=(4.5,4.5,4.5))
SetProperties(Height=11)
Show()


import glob
import re
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s
    
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

data_ = glob.glob('/home/rp269144/build/output*.vtk')
sort_nicely(data_)

simulation=OpenDataFile(data_)
display=GetDisplayProperties(simulation)
display.Representation = 'Point Gaussian'
display.GaussianRadius = 0.5
Show()
