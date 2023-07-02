import sys
sys.path.append('.../')
from MCNeuronSim import Neuron
from MCNeuronSim import Electrode

import matplotlib.animation as animation
from IPython import display




class Neuropixel_Shank:
    '''
    A geometric model of the neuropixel shank that combines electrode readout functionality with an in vitro neuron
    
    Attributes: 
    @type samp_rate: float
        sample rate [Hz]
    @type persist_len: float
        the length at which angles between tangent vectors in the neuron become uncorrelated, given in microns [um]
        
    @type num_E: int
        the number of electrodes in the neuropixel shank
    @type Electrode_1-12: Electrode
        electrode object that comprises the neuropixel shank
    @type Shank: list[Electrode, Electrode,...,Electrode]
        list containing each electrode in the neuropixel shank for easy iteration
    
    @type x: np.array
        numpy array that defines the x-coordinates of the meshgrid describing the neuropixel shank
    @type y: np.array
        numpy array that defines the y-coordinates of the meshgrid describing the neuropixel shank
    @type activity_meshgrid: np.array
        2d numpy array that shows the electrode activity at every point in the meshgrid describing the neuropixel shank
    @type time_bin_indices: list
        A list containing the indices in the Neuron's tvec array at which the voltage at each electrode would be sampled at the current samp_rate
    '''
    
    def __init__(self, samp_rate, persist_len = 1000):
        self.samp_rate = samp_rate #Hz
        self.persist_len = persist_len # um
        
        #initializing neuron placeholder, needs to be generated in separate function
        self.tissue = None #Neuron object
        
        #initializing electrode array (I hardcoded the electrode locations because the neuropixel is static)
        self.num_E = 12
        self.Electrode_1 = Electrode(self.samp_rate, (-24.6,50), "1")
        self.Electrode_2 = Electrode(self.samp_rate, (8.2,50), "2")
        self.Electrode_3 = Electrode(self.samp_rate, (-8.2,30), "3")
        self.Electrode_4 = Electrode(self.samp_rate, (24.6,30), "4")
        self.Electrode_5 = Electrode(self.samp_rate, (-24.6,10), "5")
        self.Electrode_6 = Electrode(self.samp_rate, (8.2,10), "6")
        self.Electrode_7 = Electrode(self.samp_rate, (-8.2,-10), "7")
        self.Electrode_8 = Electrode(self.samp_rate, (24.6,-10), "8")
        self.Electrode_9 = Electrode(self.samp_rate, (-24.6,-30), "9")
        self.Electrode_10 = Electrode(self.samp_rate, (8.2,-30), "10")
        self.Electrode_11 = Electrode(self.samp_rate, (-8.2,-50), "11")
        self.Electrode_12 = Electrode(self.samp_rate, (24.6,-50), "12")
        self.Shank = [self.Electrode_1, self.Electrode_2, self.Electrode_3, self.Electrode_4, self.Electrode_5, self.Electrode_6, self.Electrode_7, self.Electrode_8, self.Electrode_9, self.Electrode_10, self.Electrode_11, self.Electrode_12]
        
        #shank geometry
        self.x = np.round(np.linspace(-35, 35, int(70/0.1)+1, endpoint=True), 1)
        self.y = np.round(np.linspace(-60, 60, int(120/0.1)+1, endpoint = True), 1)
        self.activity_meshgrid = -1*np.ones((len(self.y), len(self.x))) #setting nonelectrode area to -1 activity
        
        #setup stuff for animation
        self.fig = None
        self.ax = None
        
        
        #time bin indices for monte carlo analysis
        self.time_bin_indices = []
        
        #Generate and fire Neuron upon initialization
        self.gen_neuron()
        self.fire_neuron()

    
    
    #Generate a neuron and update self.tissue with it
    def gen_neuron(self):
        #randomly generate the origin from an x-boundary, the middle two electrodes in y, and random z near the shank
        x = np.random.choice([-35, 35])
        y = np.random.uniform(-16,16)
        z = np.random.uniform(0,10)
        origin = (x, y, z) #x, y, z [um]
        bounding  = (-35, 35, -40, 40, 0)
        
        #update self.tissue with the neuron
        self.tissue = Neuron(self.samp_rate, origin, bounding, self.persist_len)
        
        #get discretization of time bins
        time_bins = np.arange(self.tissue.tvec[0], self.tissue.tvec[-1], 1/self.samp_rate )
       
        #get indices where time_bins align with time vector
        for time_bin in time_bins:
            index = np.argmin(np.abs(self.tissue.tvec-time_bin))
            self.time_bin_indices.append(index)
        
    
    
    #Set the potential at each electrode based on the location of the action potential in the neuron as a function of time
    def fire_neuron(self):
        for i in range(len(self.tissue.tvec)):
            for electrode in self.Shank:
                #cartesian position of action potential 
                X = self.tissue.x_steps[i]
                Y = self.tissue.y_steps[i]
                Z = self.tissue.z_steps[i]
                V = self.tissue.spatial_voltage[i]
                
                #logic for determining spatial fall-off of action potential
                if electrode.on_electrode(X, Y) and Z < 1: 
                    #is neuron on the electrode?
                    electrode.activity = V
                elif electrode.on_electrode(X, Y): 
                    #is neuron above the electrode?
                    electrode.activity = V/Z
                else: #neuron is adjacent to an electrode
                    x_cen, y_cen = electrode.center
                    
                    #get distance from electrode to neuron action potential
                    x_elec = np.abs(x_cen - X) 
                    y_elec = np.abs(y_cen - Y)
                    
                    #get correction from center to edge of electrode
                    theta = np.arctan(y_elec/x_elec)
                    if theta <= np.pi/4:
                        d = 6/np.cos(theta)
                    elif theta > np.pi/4:
                        d = 6/np.sin(theta)
                    
                    #Calculate R
                    R = ((x_elec - d*np.cos(theta))**2 + (y_elec - d*np.sin(theta))**2 + (Z)**2)**0.5
                    if R < 1:
                        electrode.activity = V
                    else: 
                        electrode.activity = V/R
                    
                    
                    
    #plot the potential of all the electrodes simultaneously
    def readout(self):       
        fig, axs = plt.subplots(nrows = 3, ncols = 4, figsize = (20,12))
    
        i = 0
        for ax in np.ravel(axs):
            electrode = self.Shank[i]
            electrode.readout(ax, self.tissue.tvec)
            i += 1
            
        plt.subplots_adjust(left=0.1, bottom=0.1, right=1, top=1, wspace=0.3, hspace=0.5)
        fig.suptitle('Neuropixel Shank Electrode Readout', fontweight = 'bold', fontsize=30, y = 1.10)

    
    
    #Retrieves the activity level of each electrode and updates the shank activity meshgrid with these values    
    def update_shank_activity(self, frame_num):
        for electrode in self.Shank:
            x = electrode.x
            y = electrode.y
            electrode.activity = electrode.v_readout[frame_num] #setting the activity with the location of action potential
            electrode.v_readout.pop() #removing above from the voltage readout so it doesn't cause length casting issues in readout function
            activity = electrode.activity_meshgrid
            
            #getting x/y indices in the non-flattened array
            x_indices = np.where(np.logical_and(self.x >= x[0], self.x <= x[-1]))[0]
            y_indices = np.where(np.logical_and(self.y >= y[0], self.y <= y[-1]))[0]
            
            #getting indices of the flattened array
            indices = []
            for element in y_indices:
                row = (x_indices + 1) + (element * len(self.x))
                indices.extend(row)
            indices = np.array(indices)
            
            #updating activity of the meshgrid
            np.put(self.activity_meshgrid, indices, activity)
            

            
    #contour plot of shank activity_meshgrid + 2D projection of 3D neuron atop the shank
    def plot_shank(self, frame_num):
        #obtain indices of discretized time bins
        index = frame_num*10
       
        #update shank_activity
        self.update_shank_activity(index)
        
        #Contour Plot
        self.ax.contourf(self.x, self.y, self.activity_meshgrid, vmin = -1, vmax = 1,cmap = cmap)
        self.ax.axis('scaled')
        
        #Neuron Plot
        if self.tissue is not None:
            #plot neuron
            self.ax.plot(self.tissue.x_steps, self.tissue.y_steps, color = 'red')
            
            #plot location of action potential
            self.ax.plot(self.tissue.x_steps[index], self.tissue.y_steps[index], linestyle = '', marker = 'o', color = 'cyan')
            
            #plot boundary lines
            boundary = np.ones(len(self.x))
            self.ax.plot(self.x, boundary * self.tissue.bounding[2], linestyle = 'dashed', color = "black")
            self.ax.plot(self.x, boundary * self.tissue.bounding[3], linestyle = 'dashed', color = "black")
        
        #Labels
        self.ax.set_xlabel(r"X [$\mu$m]", fontweight = "bold", fontsize = 15)
        self.ax.set_ylabel(r"Y [$\mu$m]", fontweight = "bold", fontsize = 15)
        self.ax.set_title("Electrode Activity", fontweight = "bold", fontsize = 25)
        self.ax.set_xlim(-35,35)
        plt.show()
        
        #Returning axis object as "frame"
        frame = self.ax
        
        return frame
            
        
    #Animation of action potential as it translates down the neuron, with electrode activity included
    def animate_shank(self, image = False, frame_num = 0):        
        figure, axis = plt.subplots(figsize = (7,8))
        self.fig = figure
        self.ax = axis
        #Turn plt output off so that the animation goes to the embedded html5 video
        plt.ioff()
        
        #animate and display html5 video
        anim = animation.FuncAnimation(self.fig, self.plot_shank, frames = len(self.tissue.tvec)//10, interval=50) #animate every tenth frame
        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        
        #close animation stream and reenable plt output
        plt.close()
        plt.ion()
        
        #generate still image if requested
        if image == True:
            self.plot_shank(frame_num)
        
        return anim
    
    
    
    #Reset the readings from the entire neuropixel shank
    def full_reset(self):
        #reset electrode
        for electrode in self.Shank:
            electrode.reset()
            
        #reset neurons
        self.time_bin_indices = []
    
    
    
    #determine kilosort + prop detection
    def detection(self):
        #get activity of all electrodes
        tot_activity = np.zeros(len(self.time_bin_indices))
        num_electrodes = 0
        for electrode in self.Shank:
            #for each time bin, append the activity at that point
            activity_bins = []
            for index in self.time_bin_indices:
                sample_bin = electrode.a_readout[index:index + self.tissue.resolution] 
                max_activity = max(sample_bin) 
                activity_bins.append(max_activity)
            
            num_electrodes += max(activity_bins) #if the electrode was active during any of it's bins, max(activity_bins)==1
            tot_activity += np.array(activity_bins) 
        
        #if only one electrode is excited
        if num_electrodes == 1:
            kilosort = 1
            prop = 0
            
        #More than one elctrode is excited
        elif num_electrodes > 1: 
            kilosort = 1
            
            #If the maximum simultaneous electrode activity is 1, then no electrodes overlap in time
            if max(tot_activity) == 1:
                prop = 1
            #check time bin simultaneity 
            elif max(tot_activity) > 1:
                #if the difference in magnitude between two activity time bins is one, they're detected in two time bins and prop detects it
                activity_diff = np.diff(tot_activity) 
                
                #thus if the number of indices where this occurs is greater than zero, prop detects the neuron
                if len(np.where(activity_diff == 1)[0]) > 0:
                    prop = 1
                else: 
                    prop = 0

        else:
            kilosort = 0
            prop = 0

        return kilosort, prop
    
    
    
    #The monte_carlo simulation
    def monte_carlo(self, iterations):
        kilosort = 0 
        prop = 0
        frac_p_k = []
        
        for sim in range(iterations):
            #reset the electrode array
            self.full_reset()
            
            #simulate the neuron
            self.gen_neuron()
            self.fire_neuron()
            kilo, propagation = self.detection()
            
            #update algorithm tally
            kilosort += kilo
            prop += propagation
            
            #If no neuron detected, skip fraction calculation to avoid dividing by zero errors
            if kilosort == 0:
                continue
                
            frac = prop/kilosort
            frac_p_k.append(frac)
        
        return frac_p_k
    
    
    
    #Plot the monte carlo simulation
    def plot_monte_carlo(self, iterations):
        fraction = self.monte_carlo(iterations)
        x_axis = np.array(range(len(fraction))) + 1
        
        plt.figure(figsize = (15,8))
        plt.plot(x_axis, fraction, linestyle = "", marker = ".")
        plt.xlabel("Number of Iterations", fontweight = "bold", fontsize = 15)
        plt.ylabel(r"Fraction of Prop to Kilosort", fontweight = "bold", fontsize = 15)
        plt.title(f"Monte Carlo Value: {fraction[-1]:.5f}", fontweight = "bold", fontsize = 25)
        plt.show()
