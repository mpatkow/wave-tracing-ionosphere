import numpy as np
import math
import scipy.constants
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.interpolate
import sys
import extractor
import mpl_toolkits.mplot3d.axes3d as axes3d


# GET MOST RECENT CURL FROM THE WEBSITE

earth_radius = 6371*1000
transmitter_f = 1.0*10**7                                                   # In Hz
#wave_vel = 3*10**8                                                      # in m/s
transmitter_position_spher = np.array([earth_radius,3.141*6/8, 200/180 * 3.141])
initial_direction = np.array([3*10**8,0,0.00001])                             # d/dt[r, theta, phi], where theta = polar angle, phi = azymuthal angle 
#initial_direction /= np.sqrt(np.sum(np.square(initial_direction)))      # Normalize

animation_count = 100000

#get correct velocity later


#time_step = 0.00000333333                                               # In s
time_step = 0.000000333333                                               # In s

def plasma_frequency(ne):
    return 1/(2 * scipy.constants.pi) * np.sqrt(ne * scipy.constants.elementary_charge ** 2 / (scipy.constants.epsilon_0 * scipy.constants.electron_mass))

def plot_wavefront():

    #plt.plot([wavefront_coords[0]],[wavefront_coords[1]], "or")
    #plt.arrow(wavefront_coords[0], wavefront_coords[1], wavefront_direction[0]*time_step, wavefront_direction[1]*time_step, head_width=1, head_length=1, color="red")
    pass


# going over poles could cause logical error
def update_pos_fixed(pos, direction,  step):
    next_pos = []
    next_pos.append(direction[0] * step + pos[0])
    merid = (pos[0]*direction[1] * step + pos[1]) % (2*math.pi)
    if merid > math.pi:
        merid -= math.pi
        merid = math.pi - merid
    next_pos.append(merid)
    next_pos.append((pos[0]*math.sin(pos[1])*direction[2]*step + pos[2]) % (2*math.pi))
    return np.array(next_pos)
    

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
    geo_coords.append((spher_coords[0] - earth_radius)/1000)
    geo_coords.append((scipy.constants.pi / 2 - spher_coords[1]) * 180 / (scipy.constants.pi))
    geo_coords.append(spher_coords[2] * 180 / scipy.constants.pi)
    return np.array(geo_coords)



wavefront_coords = transmitter_position_spher[:]
wavefront_direction = initial_direction[:]
wavefront_f = transmitter_f

dex = extractor.Extractor()
fname = "data-no-upload/wfs.t00z.ipe10.20230629_211000.nc"
ne = get_electron_density(dex, fname)
alt = dex.extract_data(fname, "alt")
lat = dex.extract_data(fname, "lat")
lon = dex.extract_data(fname, "lon")


XS = []
YS = []
ZS = []
c = 0
ge_coords_full_1  = []
ge_coords_full_2  = []
while True:
    c+= 1
    # STEP 1: update wavefront position

    wavefront_coords_new = update_pos_fixed(wavefront_coords, wavefront_direction, time_step)
    
    # STEP 2: test plasma freq.
    # does this need to be done every time??
    ne_interp = scipy.interpolate.RegularGridInterpolator((alt, lat, lon), ne)
    try:
        f_pe = plasma_frequency(ne_interp(convert_to_geo_coords(wavefront_coords)))
    except ValueError as e:
        f_pe = 0

    print(f_pe)

    if wavefront_f < f_pe:
        # wave is evanescent
        # FIXME
        print('reflecting')
        wavefront_direction[0] *= -1
        #wavefront_coords_new = wavefront_coords
        wavefront_coords = wavefront_coords_new 
    else:
        # wave continues
        wavefront_coords = wavefront_coords_new 



        
    plot_wavefront()
    theta, phi = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 40)
    #THETA, PHI = np.meshgrid(theta, phi)
    R = wavefront_coords[0]
    THETA = wavefront_coords[1]
    PHI = wavefront_coords[2]
    print(R)
    print(THETA * 180 / math.pi)
    print(PHI * 180 / math.pi)
    print()
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    print(X)
    print(Y)
    print(Z)
    print()
    XS.append(X)
    YS.append(Y)
    ZS.append(Z)
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1, projection='3d')
    lim = 10000000
    #ax.axes.set_xlim3d(left=-lim, right=lim) 
    #ax.axes.set_ylim3d(bottom=-lim, top=lim) 
    #ax.axes.set_zlim3d(bottom=-lim, top=lim)
    #plot = ax.scatter(np.array(XS),np.array(YS),np.array(ZS))
    # sphere plot
    r = earth_radius
    u , v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x1 = r*np.cos(u)*np.sin(v)
    y1 = r*np.sin(u)*np.sin(v)
    z1 = r*np.cos(v)
    #ax.plot_wireframe(x1, y1, z1, color="r")

    lonp = 356 
    altp = 100
    g_coords = convert_to_geo_coords(wavefront_coords)
    ge_coords_full_1.append(g_coords[0]/2655*altp)
    ge_coords_full_2.append(g_coords[2]/356*lonp)

    if c > animation_count:
        plt.show()
        lon_i = np.linspace(0,356,lonp)
        alt_i = np.linspace(0, 2655, altp)
        d = np.ones((altp, lonp))
        const_lat = convert_to_geo_coords(transmitter_position_spher)[1]
        for dy in range(altp):
            for dx in range(lonp):
                try:
                    d[dy][dx] = ne_interp(np.array([[alt_i[dy], const_lat, lon_i[dx]]]))
                except:
                    pass

        plt.imshow(d, aspect="auto", origin = "lower")
        #plt.scatter(ge_coords_full_2, ge_coords_full_1, color="r")
        plt.plot(ge_coords_full_2, ge_coords_full_1, color="r")
        plt.show()

    else:
        plt.close()


    #plt.imshow(ne[:][:][int(const_lat)],origin='upper',aspect='auto')#,extent=[0,100*1000,0,100*1000])
    #plt.imshow(g[1],origin='upper',aspect='auto',extent=[0,100*1000,0,100*1000])
    #plt.colorbar()
    #plt.show()

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
