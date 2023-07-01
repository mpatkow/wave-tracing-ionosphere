import numpy as np
import scipy.constants
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

transmitter_f = 10                  # In Hz
transmitter_position = [1,2,3]      # In [x,y,z]
initial_direction = [1,2,3]         # In [x,y,z]
time_step = 1                       # In s
simulation_steps = 100              # Num of steps whole simulation should last

def plasma_frequency(ne):
    return 1/(2 * scipy.constants.pi) * np.sqrt(ne * scipy.constants.elementary_charge ** 2 / (scipy.constants.epsilon_0 * scipy.constants.electron_mass))

def plot_wavefront():
    ax.plot([wavefront_coords[0]],[wavefront_coords[1]], "or")
    ax.arrow(wavefront_coords[0], wavefront_coords[1], vel*wavefront_direction[0], vel*wavefront_direction[1], head_width=1, head_length=1, color="red")


test_data = np.ones((100,100))
for j in range(len(test_data)):
    for i in range(len(test_data[0])):
        if j > 80:
            test_data[j][i] = 100 
test_data = np.array(test_data)

def animation_func(i):
    ax.clear()
    ax.plot([i],[i],"o")
    # add auto scale
    ax.imshow(test_data,origin='lower',aspect='auto')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])

def update_pos(pos, vel, step):
    return pos + vel*step


#make module to import noaa data

wavefront_coords = transmitter_position[:]
wavefront_direction = initial_direction[:]

# Right now we are assuming the speed is the speed of light:
#vel = 3*10**8
TEST_FREQ = 1

fig, ax = plt.subplots(1,1)

wavefront_f = transmitter_f

print(update_pos(np.array([1,2,3]), np.array([1,4,0]), 1))
sys.exit()





#######################################################################################


ani = FuncAnimation(fig, animation_func, frames=10,
                    interval=1, repeat=False)
plt.close()

from matplotlib.animation import PillowWriter
# Save the animation as an animated GIF
ani.save("simple_animation.gif", dpi=300,
         writer=PillowWriter(fps=60))
sys.exit()

while True:
    # STEP 1: update wavefront position
    
    # STEP 2: test plasma freq.
    if wavefront_f < TEST_FREQ:
        print('reflecting')
        vel *= -1
        TEST_FREQ = 1
        
    wavefront_coords = wavefront_coords_new 
    plot_wavefront()
    print(wavefront_direction)
    print(vel)
    #ax.imshow(test_data,origin='lower',aspect='auto')
    #ax.show()


    TEST_FREQ += 1

