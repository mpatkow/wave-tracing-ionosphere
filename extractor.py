import numpy as np
import datetime
import netCDF4 as nc
import requests

class Extractor:
    def __init__(self):
        self.base_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/wfs/prod/"

    def get_first_dir(self, currtime):
        return "wfs." + str(currtime.year).zfill(4) + str(currtime.month).zfill(2) + str(currtime.day).zfill(2)


    def get_full_url(self):
        fullurl = self.base_url 
        currtime = datetime.datetime.utcnow()
        first_dir = self.get_first_dir(currtime)
        fullurl += first_dir + "/"

        for hour in ["00,06,12,18"]:
            try:
                print(first_dir)
            except:
                pass


        return fullurl

    def extract_data(self, filename, k):
        data = nc.Dataset(filename)
        return data[k][:]

    def get_keys(self, filename):
        data = nc.Dataset(filename)
        print(data)
        

#e = Extractor()
#print(e.get_full_url())

#e.extract_data("wfs.t18z.ipe10.20230702_180000.nc")
