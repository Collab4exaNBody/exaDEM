import math 
import numpy
from matplotlib import pyplot as plt
from itertools import count, takewhile
from numpy import array

def frange(start, stop, step):
	return takewhile(lambda x: x< stop, count(start, step))


############## Please, enter your input parameters here  ########
## Driver Input Variables
driver_id = 0
driver_type = surface #"stl_mesh" "surface"
driver_center = array([0.0,0.0,0.0])
driver_offset = array([0.0,0.0,0.0])

driver_stl_filename = "test.stl" 

time_start = 0.
time_end = 5.
dt = 1.0e-1

### Shaker Input Variaple
amplitude = 1.0
omega = 1.0e9
directions = array([0.0,0.0,1.0]) # Z

print("Driver Id     -> " + str(driver_id))
print("Driver type   -> " + str(driver_type))
print("Driver center -> [" + str(driver_center[0]) + "," + str(driver_center[1]) + "," + str(driver_center[2]) + "] ")
print("Time          -> [" + str(time_start) + ":" + str(time_end) + "]")
print("Delta t       -> " + str(dt))

if(driver_type == "stl_mesh"): print("STL File is " + driver_stl_filename)

print()
print()
print("Please, copy these lines into your input file (.msp):")
print()
print()

################# Do not touche ################################
if(time_end <= time_start):
	print("Wrong time definition")

if(omega == 0.0):
	print("Warning, omega is equal to 0")

if(amplitude == 0.0):
	print("Warning, the amplitude is equal to 0")

def check_type(dtype):
	if(dtype == "surface"): return True
	if(dtype == "stl_mesh"): return True
	if(dtype == "ball"): return True
	if(dtype == "cylinder"): return True
	return False


def register_type(dtype):
	tmpStr = "register_" + dtype
	return tmpStr


if(not check_type(driver_type)):
	print("Wrong driver type")

## init output streams
position_values = ""
time_values = ""

## get number of timestep
size = int((time_end - time_start) / dt + 1)



if(driver_type != "surface"):
	## Init arrays to displays values
	px = numpy.zeros(size)
	py = numpy.zeros(size)
	pz = numpy.zeros(size)

	for t in range(size):
		time = float(t) * dt + time_start
		## positions
		signal = amplitude * math.sin( omega * time )
		position_value = driver_center + float(signal) * directions
		px[t] = position_value[0]
		py[t] = position_value[1]
		pz[t] = position_value[2]
		tmpStr = " [" + str(position_value[0]) + "," + str(position_value[1]) + "," + str(position_value[2]) + "] ,"
		position_values += tmpStr
		## time
		time_values += " " + str( time ) + " ,"

	time_values = time_values[:-1]
	position_values = position_values[:-1]

	print("+setup_driver:")
	print("  - " + register_type(driver_type) + ":")
	print("     id: " + str(driver_id))
	if(driver_type == "stl_mesh"): print("     filename: " + driver_stl_filename)
	print("     state: { center: [" + str(driver_center[0]) + "," + str(driver_center[1]) + "," + str(driver_center[2]) + "] }" )
	print("     params:")
	print("        motion_type: TABULATED")
	print("        time: [" + time_values +"]")
	print("        positions: [" + position_values + "]")


	## save plot
	fig, (ax, ay, az) = plt.subplots(3)
	Time = numpy.arange(0,size)
	Time = Time * dt + time_start
	ax.set_ylabel("Position X")
	ay.set_ylabel("Position Y")
	az.set_ylabel("Position Z")
	ax.set_xlabel("Time(s)")
	ay.set_xlabel("Time(s)")
	az.set_xlabel("Time(s)")
	ax.plot(Time, px)
	ay.plot(Time, py)
	az.plot(Time, pz)
	plt.savefig("shaker.png")
