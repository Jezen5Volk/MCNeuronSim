import numpy as np
import matplotlib.pyplot as plt


class Neuron: 
    '''
    A self-generating model of a neuron complete with the ability to fire an action potential. 
    It assumes that it is given a starting point on the y-axis.
    
    Attributes: 
    @type samp_rate: float
        the rate at which the neuropixel shank samples it's electrodes
    @type origin: tuple (float, float, float)
        location of the origin of neuron generation in cartesian coordinates (x, y, z) given in microns [um]
    @type bounding: tuple (float, float, float, float)
        location of the boundaries of generation (x_min, x_max, y_min, y_max, z_min) given in microns [um]
    @type persist_len: float
        the length at which angles between tangent vectors become uncorrelated, given in microns [um]
    @type resolution: float
        how many equally spaced steps will be included per sample (related to the sample rate) when growing the neuron, unitless quantity
   
    @type theta: float
        the initial center of the distribution for the azimuthal angle, given in radians [rad]
    @type phi: float
        the initial center of the distribution for the polar angle, given in radians [rad]
    @type grow: bool
        flag to determine whether the neuron should keep growing during neuron generation
    @type action_potential: float
        the strength of action potentials fired by this neuron, given in microvolts [uV]
    @type pulse_speed: float
        the speed of action potentials fired by this neuron, given in microns per second [um/sec]
    @type Lambda: float
        defining length of each step so that time bins are 100 times smaller than the sample rate, given in microns [um]
    @type direction: int
        parameter to ensure rho goes in the correct direction based on which boundary it originates
    @type x_steps: np.array
        an array containing the generated x_values of the neuron in microns [um]
    @type y_steps: np.array
        an array containing the generated y_values of the neuron in microns [um]
    @type z_steps: np.array
        an array containing the generated z_values of the neuron in microns [um]
    @type spatial_voltage: np.array
        an array containing the voltage of the action potential at each point in space due to exponential decay in microvolts [uV]
    @type decay_constant: float
        the value of the spatial decay constant of L5 pyramidal neurons, given in microns [um]
    @type tvec: np.array
        an array with the time when the action potential peak is at the corresponding x/y/z step index, given in seconds [sec]
    '''
    
    def __init__(self, samp_rate, origin, bounding, persist_len=1000, resolution = 100):
        self.samp_rate = samp_rate #Sampling rate of the neuropixel shank, [Hz]
        self.origin = origin #origin located on the y axis, [um]
        self.bounding = bounding #bounding given in [um]
        self.persist_len =  persist_len #persistence length of a neuron [um]
        self.resolution = resolution #resolution of neuron growth, unitless
        
        #derived/generated neuron properties
        self.theta = np.random.uniform(0, np.pi) # [rad]
        self.grow = True 
        self.action_potential = np.random.uniform(20,500) #pulse voltage [uV], range of spike values from recording used to generate 5rms thresholds 
        self.pulse_speed = np.random.uniform(500000,2000000) #neuron pulse speed [um/sec], doi: 10.1152/jn.00628.2014
        self.Lambda = self.pulse_speed/(100*self.samp_rate) #single step length [um]
        self.x_steps = [self.origin[0]] # [um]
        self.y_steps = [self.origin[1]] # [um]
        self.z_steps = [self.origin[2]] # [um]
        self.spatial_voltage = [self.action_potential] # [uV]
        self.decay_constant = 455 # spatial decay contstant [um], taken from: https://doi.org/10.1038/nature04720 
        self.tvec = [0] # [sec]
        
        
        #It is assumed that the origin of the neuron will be placed at either y_min or y_max on the neuron boundary
        if self.origin[0] == self.bounding[0]:
            #if placed at x_min, dx should initially be positive
            self.phi = np.random.uniform(-np.pi/2, np.pi/2) # [rad]
        elif self.origin[0] == self.bounding[1]:
            #if placed at x_max, dx should initially be negative
            self.phi = np.random.uniform(np.pi/2, 3*np.pi/2) # [rad]
        else:
            raise Exception("X coordinate of the origin of the neuron should equal either x_min or x_max in bounding")
            
        #After defining all parameters, the neuron should generate
        self.generate_neuron()

        
        
    #based on initial starting point, generate a step size and angle, then return the ending starting point
    def gen_len_step(self, x_i, y_i, z_i, t_i, v_i):
        #theta_r and phi_r are with respect to the r_hat, phi_hat, theta_hat coordinate system relating to the previous step
        theta_r =  np.random.rayleigh(scale = np.sqrt(self.Lambda/self.persist_len))
        phi_r = np.random.uniform(0, 2*np.pi)  # [rad]
        
        #to get increment in polar angles, use polar relationship between r_hat coordinate system and polar coordinate system
        dtheta = np.sin(theta_r)*np.cos(phi_r)
        dphi = np.sin(theta_r)*np.sin(phi_r)
        
        #add these angles to the previous step's polar angles
        self.theta = self.theta + dtheta
        self.phi = self.phi + dphi
         
        #convert angle increment to cartesian coordinates 
        dx = self.Lambda*np.sin(self.theta)*np.cos(self.phi)
        dy = self.Lambda*np.sin(self.theta)*np.sin(self.phi)
        dz = self.Lambda*np.cos(self.theta)
        dt = self.Lambda/self.pulse_speed
                
        #increment the neuron
        x_f = x_i + dx
        y_f = y_i + dy
        z_f = z_i + dz
        t_f = t_i + dt
        v_f = v_i*np.exp(-self.Lambda/self.decay_constant)

        
        #check bounds
        if x_f < self.bounding[0]:
            self.grow = False
        elif x_f > self.bounding[1]:
            self.grow = False
        if y_f < self.bounding[2]:
            self.grow = False
        elif y_f > self.bounding[3]:
            self.grow = False
        if z_f < self.bounding[4]:
            z_f = self.bounding[4] #this is a 2d array
        
        return x_f, y_f, z_f, t_f, v_f
    
    
    
    #generate the neuron by iteratively calling self.gen_len_step until the self.grow flag is false
    def generate_neuron(self):
        #unpacking initial points for generation from the origin
        x, y, z = self.origin
        t = 0
        v = self.action_potential
        
        while (self.grow == True):
            #use gen_len_step function to calculate next step
            x_f, y_f, z_f, t_f, v_f = self.gen_len_step(x, y, z, t, v)
            
            #appending each step to their respective lists
            self.x_steps.append(x_f)
            self.y_steps.append(y_f)
            self.z_steps.append(z_f)
            self.tvec.append(t_f)
            self.spatial_voltage.append(v_f)
            
            #reassigning for the next iteration of the loop
            x = x_f
            y = y_f
            z = z_f
            t = t_f
            v = v_f
        
        #Typecasting to np.array since it's a more useful data structure
        self.x_steps = np.array(self.x_steps)
        self.y_steps = np.array(self.y_steps)
        self.z_steps = np.array(self.z_steps)
        self.tvec = np.array(self.tvec)
        self.spatial_voltage = np.array(self.spatial_voltage)
    
    
    
    def plot_neuron(self):
        fig = plt.figure(figsize=(15,7))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot3D(self.x_steps, self.y_steps, self.z_steps, 'red') 
        ax.set_xlabel("X", fontweight= "bold")
        ax.set_ylabel("Y", fontweight= "bold")
        ax.set_xlim(self.bounding[0], self.bounding[1])
        ax.set_ylim(self.bounding[2],self.bounding[3])
        ax.view_init(90, 270)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot3D(self.x_steps, self.y_steps, self.z_steps, 'red') 
        ax.set_xlabel("X", fontweight= "bold")
        ax.set_zlabel("Z", fontweight= "bold")
        ax.set_xlim(self.bounding[0], self.bounding[1])
        ax.set_ylim(self.bounding[2],self.bounding[3])
        ax.set_zlim(self.bounding[4], 10)
        ax.view_init(0, -90)
