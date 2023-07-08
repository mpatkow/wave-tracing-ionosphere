import numpy as np
import math
import scipy.constants
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.interpolate
import sys
import extractor
import mpl_toolkits.mplot3d.axes3d as axes3d

# TODO:
# GET MOST RECENT CURL FROM THE WEBSITE
# READ REAL TIME DATA
# Ignoring curvature of the earth at the moment.
# Only launch angle at the moment, no azimuthal choice
# Therefore, we only count [x,z]
# FIXME there could be a problem with getting the data at a point. (weird indexing of arrays).
# FIXME this could be fixed by making a good get_ne function
# Take nearby snapshot for gradient?
# stupid reversing of coordinates, fix.

# All values in base SI units
#earth_radius = 6371000

# Take transmitter to be origin
#transmitter_f = 1.1*10**7
#transmitter_speed = 3*10**8
#transmitter_position_global = [0,0,0]

#initial_direction = math.radians(45)                                                  # At the moment this is launch angle FIXME

#time_step = 0.00000000333333
#animation_count = 100

#wavefront_coords = [2.5,0]
#wavefront_direction = initial_direction                                             # Ranges from 0 to 2pi
#wavefront_speed = transmitter_speed
#wavefront_f = transmitter_f

#def plasma_frequency(ne):
#    return 1/(2 * scipy.constants.pi) * np.sqrt(ne * scipy.constants.elementary_charge ** 2 / (scipy.constants.epsilon_0 * scipy.constants.electron_mass))

# ne = Sum_type (ni_type)
def get_electron_density(dex, fname):
    return dex.extract_data(fname, "O_plus_density") + dex.extract_data(fname, "H_plus_density") + dex.extract_data(fname, "He_plus_density") + dex.extract_data(fname, "N_plus_density") + dex.extract_data(fname, "NO_plus_density") + dex.extract_data(fname, "O2_plus_density") + dex.extract_data(fname, "N2_plus_density")

## use copy of list as opposed to list itself
#def update_position(r, v, dt, theta):
#    r[0] += v*math.cos(theta)*dt
#    r[1] += v*math.sin(theta)*dt
#    return r

#def D(n, ne, f):
#    d_func_num = np.sqrt(np.dot(n,n))
#    d_func_denom = np.sqrt(1-(plasma_frequency(ne)**2)/(f**2))
#    return d_func_num/d_func_denom

def D_grad(position, ne_interp, d, n, f):
    # TODO remove try except
    try:
        dDx = D(n, ne_interp(np.array([position[1], position[0]+d/2])), f) - D(n, ne_interp(np.array([position[1], position[0]-d/2])), f)
        dDz = D(n, ne_interp(np.array([position[1]+d/2, position[0]])), f) - D(n, ne_interp(np.array([position[1]-d/2, position[0]])), f)
    except ValueError:
        dDx = dDz = 0
    return np.array([dDx/d, dDz/d])
    
    

slope = 0.2
ne_grid = np.ones((100,100)) 
for i in range(100):
    for j in range(100):
        if i > 30+slope*j:
            ne_grid[i][j] = 10**11*(i-30-slope*j)
ne_interp = scipy.interpolate.RegularGridInterpolator((np.linspace(0,100,100), np.linspace(0,100,100)), ne_grid)

x_p = 10
z_p = 10
x_i = np.linspace(0,100,x_p)
z_i = np.linspace(0,100,z_p)
d = np.ones((z_p, x_p))
for dy in range(z_p):
    for dx in range(x_p):
        try:
            d[dy][dx] = plasma_frequency(ne_interp(np.array([dy*100/z_p,dx*100/x_p])))
        except:
            pass

#plt.imshow(d, aspect="auto", origin = "lower")
#plt.colorbar()
#plt.show()

#animation_num = 0
#wavefront_position_history_xs = []
#wavefront_position_history_zs = []
#while animation_num < animation_count:
#    wavefront_coords_new = update_position(wavefront_coords[:], wavefront_speed, time_step, wavefront_direction)    

#    f_pe = plasma_frequency(ne_interp(np.array([wavefront_coords_new[1], wavefront_coords_new[0]])))
#
#    print(D_grad(wavefront_coords, ne_interp, 0.1, wavefront_speed * np.array([math.cos(wavefront_direction), math.sin(wavefront_direction)]), wavefront_f))
#    # 
#
#    if wavefront_f < f_pe:
#        print('reflecting')
#        n = np.array([1,-1/slope])
#        n = n / np.sqrt(np.sum(n**2))
#
#        a = -np.arctan(slope)
#        print(math.degrees(a))
#        wavefront_direction = 2 * math.pi - 2 * a - wavefront_direction

        #wavefront_direction_vector = np.array([math.cos(wavefront_direction),math.sin(wavefront_direction)])

        #wavefront_direction_new = []
        #wavefront_direction_new.append(wavefront_direction_vector[0]-2 * wavefront_direction_vector[0] * n[0] * n[0])
        #wavefront_direction_new.append(wavefront_direction_vector[1]-2 * wavefront_direction_vector[1] * n[1] * n[1])
        #a = np.gradient(plasma_frequency(ne_grid))[0][int(wavefront_coords[1])][int(wavefront_coords[0])]
        #b = np.gradient(plasma_frequency(ne_grid))[1][int(wavefront_coords[1])][int(wavefront_coords[0])]
        #print(a)
        #print(b)
        #wavefront_direction_vector_new = np.array(wavefront_direction_new)
        #wavefront_direction = math.atan2(wavefront_direction_vector_new[1],wavefront_direction_vector_new[0])
        #print(wavefront_direction)

#    wavefront_coords = wavefront_coords_new # This could cause problems with getting stuck
#
#    wavefront_position_history_xs.append(wavefront_coords[0])
#    wavefront_position_history_zs.append(wavefront_coords[1])
#    animation_num += 1
#
#plt.plot(wavefront_position_history_xs, wavefront_position_history_zs,"o")
##plt.imshow(np.gradient(plasma_frequency(ne_grid))[1], aspect="auto", origin="lower")
##plt.imshow(d, aspect="auto", origin="lower") # show fpe, not the other one
#plt.imshow(ne_grid, aspect="auto", origin="lower") # show fpe, not the other one
#plt.colorbar()
#plt.show()

