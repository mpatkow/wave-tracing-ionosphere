import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import extractor

e = extractor.Extractor()

fname = "data-no-upload/wfs.t00z.ipe10.20230629_211000.nc"
#fname = "wfs.t00z.ipe05.20230629_211000.nc"



e.get_keys(fname)
lon = e.extract_data(fname, "lon") 
lat = e.extract_data(fname, "lat") 
alt = e.extract_data(fname, "alt") 
O_plus_density = e.extract_data(fname, "O_plus_density")
print(O_plus_density)
print(np.max(O_plus_density))
print(lon)
print(lat)
print(alt)
print(lon.shape)
print(lat.shape)
print(alt.shape)
print(O_plus_density.shape)
print(O_plus_density[0].shape)

#plt.imshow(O_plus_density[:,45,:],  origin="lower")

"""
lonp = 1000
altp = 1000
lon_i = np.linspace(0,356, lonp)
alt_i = np.linspace(90,2655, altp)

d = np.ones((altp, lonp))
print(d.shape)

O_plus_density_interp = scipy.interpolate.RegularGridInterpolator((alt, lat, lon), O_plus_density)

for dy in range(altp):
    for dx in range(lonp):
        d[dy][dx] = O_plus_density_interp(np.array([[alt_i[dy], 45, lon_i[dx]]]))


#print(O_plus_density_interp(np.array([[91.0,20.0,5.0]])))

plt.imshow(d, aspect="auto", origin="lower")

plt.colorbar()
plt.xlabel("lon")
plt.ylabel("alt")
plt.show()
"""
