import glob
import numpy as np
import pandas as pd


print('Hello, this is a small script offering basic analyses of interactions. The default analyses are :')
print(' - print profile rx, ry, rz, vx, vy, vz over time')
print('[Default Parameters]:')
print('[Filename]: dem_*')
print('[Timesteps]: 500')
print('[Dt]: 0.00005')
print("[fields]: ['rx', 'ry', 'rz', 'vx', 'vy', 'vz']")

## variable
#dt = float(1)

# Get files
files = glob.glob('dem_*')
files.sort()
nfiles = len(files)

# Define labels
labels = np.array(['rx', 'ry', 'rz', 'vx', 'vy', 'vz'])
nlabels = len(labels)
mean = np.zeros((nfiles, nlabels))

# Define time parameters
dt=float(0.00005)
timesteps=float(500)

# create xplot
rng = range(0, nfiles) 
xplot = np.zeros(nfiles)
for i  in range(0,nfiles):
	xplot[i] = i*dt*timesteps


# Compute and Get means
for i in range(0,nfiles):
	tmp = pd.read_csv(files[i], sep='\s+', skiprows = 2, names=['species', 'rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'radius', 'id'])
	data = pd.DataFrame(tmp)
	## Remove species
	data.pop(data.columns[0])
	means = data.mean()
	j = 0
	for label in labels:
		mean[i,j]=means[label]
		j = j + 1
		

# Import matplotlib stuff
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker

# Set plot parameters
fig, (ax, bx) = plt.subplots(2)
ax.set_ylabel('Mean')
ax.set_xlabel('time')
bx.set_ylabel('Mean')
bx.set_xlabel('time')

# Plot means
for j in range(0, nlabels):
	yplot = mean[:,j]
	if j < 3:
		ax.plot(xplot,yplot, label=labels[j])
	else:
		bx.plot(xplot,yplot, label=labels[j])

# Save Plot
ax.grid(True)
ax.legend()
bx.grid(True)
bx.legend()
fig.savefig("mean_r_v.pdf")
print("write mean_r_v.pdf")
