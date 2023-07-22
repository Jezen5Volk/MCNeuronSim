import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.io

#loading input channel data with x, y coordinates of the electrodes and their 5RMS thresholds in microvolts [uV]
channels = scipy.io.loadmat("channels.mat")
RMS_threshold = channels["channels"][:,2] * 0.195 #the third column of channels contains the rms threshold, multiply by 0.195 to convert to microvolts

#defining custom colormap
cmap = ListedColormap(["white","black","cyan"])


class Electrode:
    '''
    A geometric model of an electrode on the neuropixel shank
    
    Attributes:
    @type samp_rate: float
        the sample rate of the electrodes, given in [Hz]
    @type center: tuple (float, float)
        location of the center of the electrode in cartesian coordinates, given in microns [um]
    @type number: string
        gives the number of the electrode in a string for identification purposes
    
    @type rms: float
        rms cutoff voltage in microvolts [uV]    
    @type activity: int
        is zero when the electrode measures voltages below the rms threshold, is one when measuring voltages above threshold
    @type x: np.array
        numpy array that defines the x-coordinates of the meshgrid describing the electrode
    @type y: np.array
        numpy array that defines the y-coordinates of the meshgrid describing the electrode
    @type activity_meshgrid: np.array
        2d numpy array that shows the electrode activity at every point in the meshgrid describing the electrode
    @type v_readout: list
        each time the electrode's activity is updated, the voltage value is stored in this list
    @type a_readout: list
        each time the electrode's activity is updated, the activity value is stored in this list
    '''
    
    def __init__(self, samp_rate, center, number):
        self.samp_rate = samp_rate
        self._center = center #location of the center of the electrode in um
        self.number = number #numbering system for electrodes
       
        #derived/generated qualities
        self.rms = np.random.choice(RMS_threshold) #rms cutoff voltage in uV
        self._activity = 0 
        self.x = np.round(np.linspace(self._center[0]-6, self._center[0]+6, int(12/0.1) + 1, endpoint = True), 1) #Electrode dim is 12 um
        self.y = np.round(np.linspace(self._center[1]-6, self._center[1]+6, int(12/0.1) + 1, endpoint = True), 1) #Want step size of 0.1
        self.activity_meshgrid = self._activity*np.ones((len(self.x), len(self.y)))
        self.v_readout = []
        self.a_readout = []
        
    
    
    @property
    def center(self):
        return self._center
    
    
    
    @center.setter
    def center(self, center):
        self._center = center 
        
        
        
    @property
    def activity(self):
        return self._activity
    
    
    
    #set electrode activity and update voltage readout
    @activity.setter
    def activity(self, voltage):
        #set activity
        if voltage >= self.rms:
            self._activity = 1
        else: 
            self._activity = 0
            
        #append to voltage/activity readouts
        self.v_readout.append(voltage)
        self.a_readout.append(self._activity)
        self.activity_meshgrid = self._activity*np.ones((len(self.x), len(self.y))) #update activity_meshgrid
            
    
    
    #given X, Y coordinates, return boolean value indicating whether or not the coordinates lie on the array
    def on_electrode(self, x, y):
        if x > self.x[0] and x < self.x[-1] and y > self.y[0] and y < self.y[-1]:
            return True
        else: 
            return False
            
        
        
    #contour plot electrode activity
    def contour_plot(self):
        plt.contourf(self.x, self.y, self.activity_meshgrid, vmin = -1, vmax = 1, cmap = cmap)
        plt.axis('scaled')
        plt.show()



    #3D plot electrode boundary
    def plot_3D(self, ax, index = 0):
        length = len(self.x)
        z = np.zeros(length)
        l_bound = np.ones(length)*(self._center[0] - 6)
        r_bound = np.ones(length)*(self._center[0] + 6)
        b_bound = np.ones(length)*(self._center[1] - 6)
        t_bound = np.ones(length)*(self._center[1] + 6)

        if self.a_readout[index] == 0:
            activity_color = 'black'
        elif self.a_readout[index] == 1:
            activity_color = 'cyan'


        ax.plot3D(l_bound, self.y, z, c = activity_color)
        ax.plot3D(r_bound, self.y, z, c= activity_color)
        ax.plot3D(self.x, b_bound, z, c= activity_color)
        ax.plot3D(self.x, t_bound, z, c= activity_color)

        return ax

    
    
    #Plot the voltage at the electrode as a function of time
    def readout(self, ax, tvec):
        ax.set_xlabel("Time [s]", fontweight = "bold", fontsize = 15)
        ax.set_ylabel(r"Voltage [$\mu$V]", fontweight = "bold", fontsize = 15)
        ax.set_title("Electrode " + self.number , fontweight = "bold", fontsize = 25)
        
        #Vertical lines indicating when sampling occurs
        samp_lines = np.arange(tvec[0], tvec[-1], 1/self.samp_rate)
        for line in samp_lines:
            ax.axvline(x = line, c = 'red')
        
        #Horizontal line indicated voltage threshold
        threshold = np.ones(len(tvec))
        ax.plot(tvec, threshold * self.rms, linestyle = 'dashed', color = "black")
        
        #Plotting electrode reading
        subplot = ax.plot(tvec, self.v_readout)
        
        return subplot
    
    
    
    #Reset the electrode and clear it's activity and voltage reading
    def reset(self):
        self.v_readout = []
        self.a_readout = []
