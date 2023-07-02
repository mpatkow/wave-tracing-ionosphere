import numpy as np
import scipy.constants
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import extractor


earth_radius = 6371.0
transmitter_f = 10**4                                                   # In Hz
wave_vel = 3*10**8                                                      # in m/s
transmitter_position_spher = np.array([earth_radius,3.141*5/8, 110/180 * 3.141])
initial_direction = np.array([1,3.141/2,0])                             # d/dt[r, theta, phi], where theta = polar angle, phi = azymuthal angle 
initial_direction /= np.sqrt(np.sum(np.square(initial_direction)))      # Normalize


time_step = 0.00000333333                                               # In s

def plasma_frequency(ne):
    return 1/(2 * scipy.constants.pi) * np.sqrt(ne * scipy.constants.elementary_charge ** 2 / (scipy.constants.epsilon_0 * scipy.constants.electron_mass))

def plot_wavefront():
    plt.plot([wavefront_coords[0]],[wavefront_coords[1]], "or")
    plt.arrow(wavefront_coords[0], wavefront_coords[1], wave_vel*wavefront_direction[0]*time_step, wave_vel*wavefront_direction[1]*time_step, head_width=1, head_length=1, color="red")

def update_pos(pos, vel, step):
    return pos + vel*step

def update_pos_fixed(pos, direction, vel, step):
    pass

# location given in (alt, lat, lon)
def get_param(location, param):
    pass

def fpe_grad(location, n):
    return np.gradient(n) 

#make module to import noaa data

# FIRST GET DATA FROM NOAA. + extrapolate.
def extrapolate():
    pass

# ne = Sum_type (ni_type)
def get_electron_density(dex, fname):
    return dex.extract_data(fname, "O_plus_density") + dex.extract_data(fname, "H_plus_density") + dex.extract_data(fname, "He_plus_density") + dex.extract_data(fname, "N_plus_density") + dex.extract_data(fname, "NO_plus_density") + dex.extract_data(fname, "O2_plus_density") + dex.extract_data(fname, "N2_plus_density")

# spher_coords = [r, theta, phi]
def convert_to_geo_coords(spher_coords):
    geo_coords = []                     # Will be in format [alt, lat, lon]
    geo_coords.append(spher_coords[0] - earth_radius)
    geo_coords.append((scipy.constants.pi / 2 - spher_coords[1]) * 180 / (scipy.constants.pi))
    geo_coords.append(spher_coords[2] * 180 / scipy.constants.pi)
    return geo_coords



wavefront_coords = transmitter_position[:]
wavefront_direction = initial_direction[:]
wavefront_f = transmitter_f

dex = extractor.Extractor()
fname = "data-no-upload/wfs.t00z.ipe10.20230629_211000.nc"
ne = get_electron_density(dex, fname)

while True:
    # STEP 1: update wavefront position
    wavefront_coords_new = update_pos(wavefront_coords, wave_vel * wavefront_direction, time_step)
    
    # STEP 2: test plasma freq.

    # 2a: get plasma frequency at that point.
    # TODO pretty sure this needs extrapolation
    f_pe = plasma_frequency(ne[wavefront_coords_new[0]][wavefront_coords_new[1]][wavefront_coords_new[2]])

    if wavefront_f < f_pe:
        # wave is evanescent
        # FIXME
        print('reflecting')
        wave_vel *= -1
        wavefront_coords_new = wavefront_coords
    else:
        # wave continues
        wavefront_coords = wavefront_coords_new 


        
    plot_wavefront()
    plt.imshow(test_data,origin='upper',aspect='auto',extent=[0,100*1000,0,100*1000])
    #plt.imshow(g[1],origin='upper',aspect='auto',extent=[0,100*1000,0,100*1000])
    plt.colorbar()
    plt.show()

# are we sure about sphericals? most likely yes.

#ani = FuncAnimation(fig, animation_func, frames=10,
#                    interval=1, repeat=False)
#plt.close()
#
#from matplotlib.animation import PillowWriter
## Save the animation as an animated GIF
#ani.save("simple_animation.gif", dpi=300,
#         writer=PillowWriter(fps=60))
#sys.exit()

#test_data = np.ones((100,100))
#for j in range(len(test_data)):
#    for i in range(len(test_data[0])):
#        if j > 30+i:
#            test_data[j][i] = 100 
#test_data = np.rot90(np.rot90(np.array(test_data)))
#transmitter_position = np.array([0,-7.6145,110.7122])           # [alt, lat, lon]
