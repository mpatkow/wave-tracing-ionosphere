import numpy as np
import datetime
import netCDF4 as nc

class Extractor:
    def __init__(self):
        self.base_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/wfs/prod/"

    def get_first_dir(self, currtime):
        return "wfs." + str(currtime.year).zfill(4) + str(currtime.month).zfill(2) + str(currtime.day).zfill(2)


    def get_full_url(self):
        fullurl = self.base_url 
        currtime = datetime.datetime.now()
        fullurl += self.get_first_dir(currtime) + "/"

        return fullurl

    def extract_data(self, filename):
        data = nc.Dataset(filename)
        print(data["alt"][:].shape)
        print(data["lon"][:].shape)
        print(data["electron_temperature"][:].shape)
        

e = Extractor()
#print(e.get_first_dir(datetime.datetime.now()))
e.extract_data("wfs.t18z.ipe10.20230702_180000.nc")
