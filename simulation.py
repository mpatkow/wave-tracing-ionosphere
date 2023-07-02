import numpy as np
import scipy.constants
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

transmitter_f = 10**4               # In Hz
transmitter_position = np.array([1,2,3])      # In [x,y,z]
initial_direction = np.array([1,2,3])         # In [vx,vy,vz]
time_step = 0.00000333333           # In s
simulation_steps = 100              # Num of steps whole simulation should last
wave_vel = 3*10**8                  # in m/s



# TODO
# make function TO CONVERT FROM INDEXES TO LON/LAT/ALT

def plasma_frequency(ne):
    return 1/(2 * scipy.constants.pi) * np.sqrt(ne * scipy.constants.elementary_charge ** 2 / (scipy.constants.epsilon_0 * scipy.constants.electron_mass))

def plot_wavefront():
    plt.plot([wavefront_coords[0]],[wavefront_coords[1]], "or")
    plt.arrow(wavefront_coords[0], wavefront_coords[1], wave_vel*wavefront_direction[0]*time_step, wave_vel*wavefront_direction[1]*time_step, head_width=1, head_length=1, color="red")


test_data = np.ones((100,100))
for j in range(len(test_data)):
    for i in range(len(test_data[0])):
        if j > 80:
            test_data[j][i] = 100 
test_data = np.rot90(np.rot90(np.array(test_data)))

def animation_func(i):
    ax.clear()
    ax.plot([i],[i],"o")
    # add auto scale
    ax.imshow(test_data,origin='lower',aspect='auto')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])

def update_pos(pos, vel, step):
    return pos + vel*step


# locatoin given in (alt, lat, lon)
def get_param(location, param):
    pass


#make module to import noaa data

# FIRST GET DATA FROM NOAA. + extrapolate.
def extrapolate():
    pass

wavefront_coords = transmitter_position[:]
wavefront_direction = initial_direction[:]

# Right now we are assuming the speed is the speed of light:
TEST_FREQ = 10


wavefront_f = transmitter_f
print(test_data)


while True:
    # STEP 1: update wavefront position
    wavefront_coords_new = update_pos(wavefront_coords, wave_vel * wavefront_direction, time_step)
    
    # STEP 2: test plasma freq.
    ycoor = 99-int(wavefront_coords_new[1]/1000)
    xcoor = int(wavefront_coords_new[0]/1000)

    freq = plasma_frequency(test_data[ycoor][xcoor])

    if TEST_FREQ < freq:
        print('reflecting')
        wave_vel *= -1
        wavefront_coords_new = wavefront_coords

        
    wavefront_coords = wavefront_coords_new 
    plot_wavefront()
    plt.imshow(test_data,origin='upper',aspect='auto',extent=[0,100*1000,0,100*1000])
    plt.show()


#ani = FuncAnimation(fig, animation_func, frames=10,
#                    interval=1, repeat=False)
#plt.close()
#
#from matplotlib.animation import PillowWriter
## Save the animation as an animated GIF
#ani.save("simple_animation.gif", dpi=300,
#         writer=PillowWriter(fps=60))
#sys.exit()

