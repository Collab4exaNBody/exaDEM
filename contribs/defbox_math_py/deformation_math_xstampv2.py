#! /usr/bin/env python
import sys, string
from math import *
import random
import numpy as np

def defbox_volume(lenghts, angles):

    al = pi*angles[0]/180.
    bt = pi*angles[1]/180.
    gm = pi*angles[2]/180.    

    vol = sqrt(1. - cos(al) * cos(al) - cos(bt) * cos(bt) - cos(gm) * cos(gm) + 2. * cos(al) * cos(bt) * cos(gm)) * lengths[0] *  lengths[1] *  lengths[2]

    return vol

def angles_to_matrix(angles):

    al = pi*angles[0]/180.
    bt = pi*angles[1]/180.
    gm = pi*angles[2]/180.    

    n2 = ( cos(al) - cos(gm) * cos(bt) ) / sin(gm)
    n3 = sqrt( sin(bt) * sin(bt) - n2 * n2)

    pm = np.array([[ 1., cos(gm), cos(bt)], 
                   [  0, sin(gm),      n2], 
                   [  0,       0,      n3]])

    return pm

def diag_matrix(lengths):

    pm = np.multiply(np.eye(3),lengths)

    return pm

    
def deformation_to_matrix(lengths,angles):

    pm0 = angles_to_matrix(angles)
    pm1 = diag_matrix(lengths)
    pm = np.dot(pm0,pm1)

    return pm

##############################################################
# Read the data, compute some data, write the data

f_input = open(sys.argv[1], "r")
f_output = open("lengths_angles_xform_volume.dat", "w")

f_output.write("# a0 \tb0 \tc0 \talpha \tbeta \tgamma \txform.m11 \txform.m12 \txform.m13 \txform.m21 \txform.m22 \txform.m23 \txform.m31 \txform.m32 \txform.m33 \tvolume \n")

while 1:
    ligne = f_input.readline()

    if len(ligne) == 0:
        break

    vals = ligne.split()

    if vals[0] == "#":
        continue
    
    if len(vals) > 0:
        lengths = np.zeros((3))
        angles = np.zeros((3))
        lengths[0] = float(vals[0])
        lengths[1] = float(vals[1])
        lengths[2] = float(vals[2])        
        angles[0] = float(vals[3])
        angles[1] = float(vals[4])
        angles[2] = float(vals[5])

        volume = defbox_volume(lengths,angles)

        def_matrix = deformation_to_matrix(lengths,angles)

        f_output.write("%5.4f \t%5.4f \t%5.4f \t%5.4f \t%5.4f \t%5.4f \t%5.4f \t%5.4f \t%5.4f \t%5.4f \t%5.4f \t%5.4f \t%5.4f \t%5.4f \t%5.4f \t%5.4f \n"%(lengths[0],lengths[1],lengths[2],angles[0],angles[1],angles[2],def_matrix[0,0],def_matrix[0,1],def_matrix[0,2],def_matrix[1,0],def_matrix[1,1],def_matrix[1,2],def_matrix[2,0],def_matrix[2,1],def_matrix[2,2],volume))
        
#        volume_bis = np.linalg.det(def_matrix)
#        print(volume_bis)


f_input.close()
f_output.close()
##############################################################
