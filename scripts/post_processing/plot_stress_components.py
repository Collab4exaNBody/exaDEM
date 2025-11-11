import os
import numpy as np
import pandas as pd
from mathutils import Vector
import argparse

print('Hello, this is a small script offering basic analyses of interactions. The default analyses are :')
print(' Usage example: python3 stress.py --directory FT4OutputDir/ExaDEMAnalyses/')

basename = 'InteractionOutputDir-'
parser = argparse.ArgumentParser(description='Script so useful.')
parser.add_argument("--directory", default="")
parser.add_argument("--volume", default=1.0)
args = parser.parse_args()
directory = args.directory
volume = args.volume

print("Scan directory                = " + directory)
print("Volume is set to              = " + str(volume))

## Get files
dirs = [d for d in os.listdir(directory)
        if basename in d and os.path.isdir(os.path.join(directory, d))]
ntimesteps = len(dirs)
print("The number of directories is  = ", ntimesteps)
	
## Get timestep
timesteps = [int(x.removeprefix(basename)) for x in dirs]
timesteps.sort()
xaxis = range(0, len(timesteps)) 
stress = np.zeros((9, ntimesteps))
xplot = timesteps

for i in xaxis:
	dirname = directory + "/" + basename + str(timesteps[i])
	files = list(os.walk(dirname))[0][2]
	## clean current data frame
	# nope
	## two first information are useless
	for filename in files:
		file_path = dirname + "/" + filename
		file_size = os.stat(file_path).st_size
		if(file_size > 0):
			tmp = pd.read_csv(file_path, sep=',',header=None)
			data = pd.DataFrame(tmp)
#			for col in range(0, 14):
#				data[col] = pd.to_numeric(data[col], errors='coerce')
			ctype = data[4]
			cpx = data[6]
			cpy = data[7]
			cpz = data[8]
			fnx = data[9]
			fny = data[10]
			fnz = data[11]
			ftx = data[12]
			fty = data[13]
			ftz = data[14]
			rx = data[15]
			ry = data[16]
			rz = data[17]
			f_vectors = [Vector((fnx[j] + ftx[j], fny[j] + fty[j], fnz[j] + ftz[j])) for j in range(len(data))]
			d_vectors = [Vector((cpx[j] - rx[j], cpy[j] - ry[j], cpz[i] - rz[j])) for j in range(len(data))]
			min_z = rz[0]
			max_z = rz[0]
			for j in	range(len(data)):
				min_z = min(min_z, rz[j] - 0.5)
				max_z = max(max_z, rz[j] + 0.5)
				if(ctype[j] >= 4 and ctype[j] < 13): #drivers
					stress[0, i] += f_vectors[j][0] * d_vectors[j][0]	# σ_xx
					stress[1, i] += f_vectors[j][1] * d_vectors[j][1]	# σ_yy
					stress[2, i] += f_vectors[j][2] * d_vectors[j][2]	# σ_zz
					stress[3, i] += f_vectors[j][0] * d_vectors[j][1]	# σ_xy
					stress[4, i] += f_vectors[j][0] * d_vectors[j][2]	# σ_xz
					stress[5, i] += f_vectors[j][1] * d_vectors[j][2]	# σ_yz
					stress[6, i] += f_vectors[j][1] * d_vectors[j][0]	# σ_yx
					stress[7, i] += f_vectors[j][2] * d_vectors[j][0]	# σ_zx
					stress[8, i] += f_vectors[j][2] * d_vectors[j][1]	# σ_zy
			# cellule ft4 -> cylinder
			for j in range(0,9):
				stress[j, i] /= volume
			print("file: " + str(i) + "/" + str(len(xaxis)) + " done." )

		
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker


fig, ax = plt.subplots()
ax.set_ylabel('Stress')
ax.set_xlabel('timesteps')


def plot_stress_components(stress: np.ndarray, time: np.ndarray):
		labels = [
				r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{zz}$",
				r"$\sigma_{xy}$", r"$\sigma_{xz}$", r"$\sigma_{yz}$",
				r"$\sigma_{yx}$", r"$\sigma_{zx}$", r"$\sigma_{zy}$"
		]

		fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
		axes = axes.ravel()

		for i in range(9):
				axes[i].plot(time, stress[i, :], lw=1.8)
				axes[i].set_title(labels[i])
				axes[i].set_xlabel("Timestep")
				axes[i].set_ylabel("Stress")
				axes[i].grid(True)

		plt.tight_layout()
		fig.savefig("stress.png")
		print("write stress.png")


plot_stress_components(stress, xplot)

