import os
import numpy as np
import pandas as pd
from collections import Counter


print('Hello, this is a small script offering basic analyses of interactions. The default analyses are :')
print(' - plot the number of interactions per types in function of the timestep (types.pdf)')
print(' - plot the number of interactions in function of the timestep (count.pdf)')

## variable
#dt = float(1)

ntypes = int(13)
basename = 'Interaction_'
directories = list(os.walk('.'))[0][1]

## Get files
dirs = [x for x in directories if 'Interaction' in x]
ntimesteps = len(dirs)
print("The number of directories is:", ntimesteps)
	
## Get timestep
timesteps = [int(x.removeprefix(basename)) for x in dirs]
timesteps.sort()
xaxis = range(0, len(timesteps)) 


## Init some variables
max_interaction = 0
mean_interaction = 0
tot_interaction = 0

types = np.zeros((ntypes, ntimesteps))
check_type = np.zeros((ntypes))

for i in xaxis:
	dirname = 'Interaction_' + str(timesteps[i])
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
			ctype = data[4]
			count = Counter(ctype)
			for t in range(0, ntypes):
				if(count.get(t)):				
					types[t, i] = types[t, i] + count.get(t)
					check_type[t]=1
#		else:
#			print("skip file (no data):", file_path)
		
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker


fig, ax = plt.subplots()
ax.set_ylabel('#interactions')
ax.set_xlabel('timesteps')

labels = np.array([
'vertex-vertex', 
'vertex-edge', 
'vertex-face', 
'edge-edge', 
'vertex-cylinder', 
'vertex-surface', 
'vertex-ball', 
'vertex-vertex(STL)', 
'vertex-edge(STL)', 
'vertex-face(STL)', 
'edge-edge(STL)', 
'vertex(STL)-edge', 
'vertex(STL)-face' ])

xplot = timesteps

### Display interactions
for t in range(0, ntypes):
	if(check_type[t] == 1):
		yplot = types[t,:]
		ax.plot(xplot,yplot, label=labels[t])
	
plt.grid()
plt.legend()
fig.savefig("types.pdf")
print("write type.pdf")

plt.clf()
fig, ax = plt.subplots()

### Plot the number of interactions
for t in range(0, ntypes):
	if(check_type[t] == 1):
		yplot = yplot + types[t,:]

ax.set_ylabel('#interactions')
ax.set_xlabel('timesteps')
ax.plot(xplot,yplot)
plt.grid()
fig.savefig("count.pdf")
print("write count.pdf")
