# importing os module
import os, sys, getopt, glob

import argparse

CheckPointDir = "/CheckpointFiles/" 

parser = argparse.ArgumentParser(description='Script so useful.')
parser.add_argument("--directory", default="ExaDEMOutputDir")

args = parser.parse_args()

Directory = args.directory
print("Restart directory: " + Directory)

FullCheckPointDir=Directory+CheckPointDir

isExist = os.path.exists(FullCheckPointDir)
if not isExist:
	print("[Error] There is no folder available for making restart templates.")
	exit()

ShapeFile = FullCheckPointDir + "RestartShapeFile.shp"
isPolyhedra = os.path.exists(ShapeFile)

if isPolyhedra:
	print("The folder contains a shape file, so polyhedral mode is enabled.")
	print("Particle mode: Polyhedra")
else:
	print("Particle mode: Spheres")


last_ite = -1 # last iteration

files = glob.glob(FullCheckPointDir + "exadem_*")
for f in files:
	f=f.replace(FullCheckPointDir,"")
	f=f.replace('exadem_','')
	f=f.replace('.dump','')
	res = int(f)
	last_ite = max(last_ite, res)


if last_ite == -1:
	print("[Error] The lastest iteration is not identified.")
	exit()
else:
	print("Last iteration identified: " + str(last_ite))

exadem_file = FullCheckPointDir + "exadem_{:010d}.dump"
exadem_file = exadem_file.format(last_ite)
driver_file = FullCheckPointDir + "driver_{:010d}.msp"
driver_file = driver_file.format(last_ite)

isDriver = os.path.exists(driver_file)

# includes
## particle type
## drivers
print("Here s a template for restarting your simulation at the last saved iteration: \n")

print("includes:")
if isPolyhedra:
	print("  - config_polyhedra.msp")
else:
	print("  - config_spheres.msp")
if isDriver:
	print("  - " + driver_file)

print("")

# input data
## shape
## restart file

print("input_data:")
print("  - init_rcb_grid")

if isPolyhedra:
	print("  - read_shape_file:")
	print("     filename: " + ShapeFile)

print("  - read_dump_particle_interaction:")
print("     filename: " + exadem_file) 

if isPolyhedra:
	print("  - radius_from_shape")

