import numpy as np
import scipy.constants
import csv
import math
import matplotlib.pyplot as plt

class raytracer:
    def __init__(self):
        pass

    def extract_data_from_conditions_file(self,filename):
        with open(filename) as conditions_csv:
            conditions_reader = csv.reader(conditions_csv, delimiter='=')
            conditions_dict = {}
            for row in conditions_reader:
                conditions_dict[row[0]] = row[1]

        conditions_dict["animation_count"] = int(conditions_dict["animation_count"])
        conditions_dict["time_step"] = float(conditions_dict["time_step"])
        conditions_dict["frequency"] = float(conditions_dict["frequency"])
        conditions_dict["position"] = [float(i) for i in conditions_dict["position"].strip('][').split(',')]
        conditions_dict["wavevector"] = [float(i) for i in conditions_dict["wavevector"].strip('][').split(',')]

        return conditions_dict 

    def fpe(self, ne):
        return 1/(2 * scipy.constants.pi) * np.sqrt(ne * scipy.constants.elementary_charge ** 2 / (scipy.constants.epsilon_0 * scipy.constants.electron_mass))

    def D(self, wave_r, wave_k, f):
        n_vec = scipy.constants.speed_of_light / (2 * scipy.constants.pi * f) * wave_k
        d_func_num = np.sqrt(np.dot(n_vec,n_vec))
        print(self.gnealc(wave_r))
        print(self.fpe(self.gnealc(wave_r)))
        d_func_denom = np.sqrt(1-(self.fpe(self.gnealc(wave_r))**2)/(f**2))
        print("HERE")
        print(d_func_num)
        print(d_func_denom)
        return d_func_num/d_func_denom
    
    def D_n_grad(self, wave_r, wave_k, f):
        n_vec = scipy.constants.speed_of_light / (2 * scipy.constants.pi * f) * wave_k
        div = np.sqrt(np.dot(n_vec,n_vec))
        gx = n_vec[0] / div
        gy = n_vec[1] / div
        gz = n_vec[2] / div
        return (1/np.sqrt(1-(self.fpe(self.gnealc(wave_r))**2)/(f**2)))*np.array([gx,gy,gz])

    def D_pos_grad(self, wave_r, wave_k, f, dr):
        dDx = self.D(wave_r+[dr,0,0], wave_k, f) - self.D(wave_r-[dr,0,0], wave_k, f)
        dDy = self.D(wave_r+[0,dr,0], wave_k, f) - self.D(wave_r-[0,dr,0], wave_k, f)
        dDz = self.D(wave_r+[0,0,dr], wave_k, f) - self.D(wave_r-[0,0,dr], wave_k, f)
        print(dDx)
        print(dDy)
        print(dDz)
        return np.array([dDx/(2*dr), dDy/(2*dr), dDz/(2*dr)])

    # TRASH CODE FIXME
    def run(self, conditions_file_name):
        """
        ne_grid = np.ones((100,100,100)) 
        for i in range(100):
            for j in range(100):
                for k in range(100):
                    if i > 30+slope*j:
                        ne_grid[i][j][k] = 10**3*(5*math.sin((i-30-slope*j)*1.1)+i-30-slope*j + 1.002**(i-30-slope*j))
        

        # DONT MAKE THIS INSTANCE VARIABLE FIXME
        self.ne_interp = scipy.interpolate.RegularGridInterpolator((np.linspace(0,100,100), np.linspace(0,100,100), np.linspace(0,100,100)), ne_grid)
        """

        ne_grid = np.ones((100,100)) 
        """
        slope = 0.3
        for i in range(100):
            for j in range(100):
                    if i > 30+slope*j:
                        ne_grid[i][j] = 10**3*(5*math.sin((i-30-slope*j)*1.1)+i-30-slope*j + 1.002**(i-30-slope*j))
                        """
        
        slope = 0.01
        for i in range(100):
            for j in range(100):
                if i > 30:
                        ne_grid[i][j] = 10**3*(i-30)

        # DONT MAKE THIS INSTANCE VARIABLE FIXME
        self.ne_interp = scipy.interpolate.RegularGridInterpolator((np.linspace(0,100,100), np.linspace(0,100,100)), ne_grid)


        conditions = self.extract_data_from_conditions_file(conditions_file_name)

        animation_num = 0
        wave_r = np.array(conditions["position"][:])
        wave_k = np.array(conditions["wavevector"][:])
        dt = conditions["time_step"]
        f = conditions["frequency"]


        wave_r_history = []

        while animation_num < conditions["animation_count"]:
            wave_r, wave_k = self.animation_step(wave_r, wave_k, dt, f)

            wave_r_history.append(wave_r)
            animation_num += 1

        wave_r_history = np.array(wave_r_history)


        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(wave_r_history[:,0], wave_r_history[:,1], wave_r_history[:,2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        """
        plt.scatter(wave_r_history[:,0], wave_r_history[:,1])
        plt.imshow(ne_grid, origin="lower", aspect="auto")
        plt.colorbar()
        plt.xlim((0,100))
        plt.ylim((0,100))
        plt.show()
    
    def animation_step(self, wave_r, wave_k, dt, f):
        # find vector derivatives
        drdt = self.D_n_grad(wave_r, wave_k, f)
        dndt = - scipy.constants.speed_of_light * self.D_pos_grad(wave_r, wave_k, f, 0.1)
        dkdt = (2*scipy.constants.pi*f)/(scipy.constants.speed_of_light) * dndt

        # update wave properties based on these derivatives
        print(wave_k)
        print(dkdt)
        print(dt)
        #input()
        wave_r = wave_r + drdt * dt
        wave_k = wave_k + dkdt * dt
        print(wave_k)
        #input()

        return wave_r, wave_k



    # Shorthand for Get NE At Local Coordinates
    # Takes coords = [x,y,z] 
    # TODO should be in separate file
    def gnealc(self,coords): 
        try:
            return self.ne_interp(np.array([coords[1], coords[0]]))[0]
            #return self.ne_interp(np.array([coords[1], coords[0], coords[2]]))[0]
        except:
            return 0

rt = raytracer()
rt.run("conditions.csv")

