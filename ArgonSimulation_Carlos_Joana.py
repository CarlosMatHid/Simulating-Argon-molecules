"""

###############################################################################
#              COMPUTATIONAL PHYSICS - LEIDEN UNIV. SPRING 2018               #
###############################################################################

    This code runs a simulation of the dynamics of a system of 108 atoms 
    interacting through Lennard-Jones potential in a cube with periodic
    boundary conditions.
    
    
    The initial conditions we set:
    - Positions: fcc lattice
    - Velocity: Gaussian distribution with a certain velocity
    
    To be fixed in the beginning of the program:
        Density and temperature: this will determine the phase of the material.
        Delta_r: the width of the spherical shell when counting particles for the 
                correlation function.
        Bootstrap_iterations: Number of resampling to compute the error.
        

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

#Defining some variables
''' gas: T=3, density = 0.3 // solid: T=0.5, density=1.2 // liquid: T=0.8, density=1'''
density = 0.3#This is the density of our box
h = 0.005 #time step
temperature = 3 #temperature
bootstrap_iterations = 50
delta_r = 0.05 #correlation function
print('Box of particles with density', density, 'and temperature', temperature, '\nPlease wait for the system to thermalize\n')


#FCC lattice Initial Conditions
N1 = 9*6
N2 = 9*6
N = N1+N2
size = (N/density)**(1/3)
position = np.zeros((N,3))
for i in np.arange(0, N1, 2):
    position[i][0] = (i%3 + 0.5)/3*size
    position[i][1] = (i//3 + 0.5)/3*size
    position[i][2] = (i//9 + 0.5)/3*size

    position[i+1][0] = (i%3 + 1)/3*size
    position[i+1][1] = (i//3 + 1)/3*size
    position[i+1][2] = (i//9 + 0.5)/3*size

for i in np.arange(N1, N1+N2, 2):
    position[i][0] = (i%3 + 1)/3*size
    position[i][1] = (i//3 + 0.5)/3*size
    position[i][2] = (i//9 + 1)/3*size

    position[i+1][0] = (i%3 + 0.5)/3*size
    position[i+1][1] = (i//3 + 1)/3*size
    position[i+1][2] = (i//9 + 1)/3*size

velocity = np.random.randn(N,3)*np.sqrt(temperature)         
force = np.zeros((N,3))

#Figure and subplots
fig = plt.figure()
ax1 = fig.add_subplot(121,aspect='equal', projection='3d')
ax2 = fig.add_subplot(422)
ax3 = fig.add_subplot(424)
ax4 = fig.add_subplot(426)
ax5 = fig.add_subplot(428)


particles, = ax1.plot(position[:,0], position[:,1], position[:,2], 'bo', ms=10)
Energy_plot, = ax2.plot([], [], lw = 2, c='b', label = 'Total energy')
T_plot, = ax2.plot([], [], lw = 2, c='r', label = 'Kinetic energy')
U_plot, = ax2.plot([], [], lw = 2, c='g', label = 'Potential energy')
Pressure_plot, = ax3.plot([], [], lw = 2, c='g', label = 'Beta*Pressure/Density')
Pressure_cum_plot, = ax3.plot([], [], lw = 2, c='r', label = 'Beta*Pressure/Density average')
Specific_heat_plot, = ax4.plot([], [], lw = 2, c='b', label = 'Specific heat')
Pair_corr_plot, = ax5.plot([], [], lw = 2, c='r', label = 'Pair correlation histogram')

ax1.set_xlim([0, size])
ax1.set_ylim([0, size])
ax1.set_zlim([0, size])
ax1.set_title('Particle box with density %0.2f \n and temperature %0.2f' % (density, temperature))
ax2.set_xlabel('time')
ax2.set_xlim([0, 6])
ax2.set_ylim([-25, 25])
ax2.grid()
ax2.legend()
ax3.set_xlabel('time')
ax3.set_xlim([0, 6])
ax3.set_ylim([0, 2])
ax3.grid()
ax3.legend()
ax4.set_xlabel('time')
ax4.set_xlim([0, 6])
ax4.set_ylim([0,10])
ax4.grid()
ax4.legend()
ax5.set_xlim([0, size/2])
ax5.set_ylim([0, 10])
ax5.legend()
ax5.grid()

#FUNCTIONS:
#Force function
def U_and_force(position):
    """takes the position of the particles (as a matrix) and returns the potential
    forces acting on every particle and the total potential energy, U   
    """
    U=0
    force=np.zeros((N,3))
    for i in range(N):
        dist_vect = np.zeros((N,3))
        dist_norm = 0
        for j in range(N):
            if i!=j:
                dist_vect[i,:]=(position[i,:]-position[j,:]+size/2)%size-size/2
                dist_norm = np.linalg.norm(dist_vect[i,:])
                force[i,:] += 24*dist_vect[i]*(2*dist_norm**(-14)-dist_norm**(-8))
                U += (4*(dist_norm**(-12)-dist_norm**(-6)))/2       

    return U, force

#Error function
def bootstrap(values, iterations):
    """Takes the an array of values and uses the bootstrap technique to compute the error of the average                   
    of them
    """
    parameter = np.average(values)
    for i in range(iterations):
        length = len(values)
        resample = np.floor(np.random.rand(length)*length).astype(int)
        parameter = np.hstack((parameter, np.average(values[resample])))
    error = np.sqrt(np.average(parameter*parameter)-np.average(parameter)**2)
    return error
     
#Correlation function
def corr_function(position, size, delta_r):
    """Computes the correlation function data to build an histogram using delta_r spacing for the bins
    """
    bins = np.linspace(0,size/2,size/2/delta_r)
    number = np.zeros(len(bins))
    for i in range(N):
        for j in range(i+1,N):
            distance= np.linalg.norm((position[i,:]-position[j,:]+size/2)%size-size/2)
            for k in range(len(bins)-1):
                if distance>=bins[k] and distance<bins[k+1]:
                    number[k]+=1

    g = 2*size**3*number/(N*(N-1)*4*np.pi*delta_r*(bins+delta_r/2)**2)  
    return g, bins
    
#Pressure
def compute_pressure(position, temperature, N):
    """Computes the pressure for given positions
    """
    sum_ = 0
    for i in range(N):
        for j in range(N):
            if j != i:
                distance= np.linalg.norm((position[i,:]-position[j,:]+size/2)%size-size/2)
                sum_ += distance*(-12*distance**(-13)+6*distance**(-7))
    pressure = 1-1/(3*N*temperature)*sum_
    return pressure
    
    
#Initialize force
U, force = U_and_force(position)

##Thermalize system
for i in range(10):
    for j in range(30):
        #Verlet computation
        vel_ = velocity + h*force/2
        position += h*vel_
        position = position%size
        U, force = U_and_force(position)
        velocity = vel_ + h*force/2

    lambda_ = np.sqrt((N+1)*3*temperature/sum(np.linalg.norm(velocity, axis=1)**2))
    print('lambda = ', lambda_)
    velocity = lambda_*velocity
   
#Initialize variables that will be plotted
time = np.array([0])
T = sum(np.linalg.norm(velocity, axis=1)**2)/2
kinetic_energy = np.array([T])
energy = np.array([U+T])
pot_energy = np.array([U])
pressure_value = compute_pressure(position, temperature, N)
pressure = np.array([pressure_value])
pressure_cum = np.array([pressure_value])
specific_heat_value = (2/3-N*(np.average(kinetic_energy**2)/np.average(kinetic_energy)**2-1))**(-1)
specific_heat = np.array([specific_heat_value])
error_pressure = 0
error_specific_heat = 0

#Animation iteration    
def iteration(i):
 
    global position, particles, velocity, h, N, size, delta_r, temperature, density, force, time, energy, kinetic_energy, pot_energy, pressure, pressure_cum, specific_heat, pair_corr, error_pressure, error_specific_heat
    
    #Verlet computation
    vel_ = velocity + h*force/2
    position += h*vel_
    position = position%size
    U, force = U_and_force(position)
    velocity = vel_ + h*force/2

    #Define array parameters that we are going to plot
    T = sum(np.linalg.norm(velocity, axis=1)**2)/2
    time = np.append(time,time[-1]+h)
    energy = np.append(energy,U+T)
    kinetic_energy = np.append(kinetic_energy,T)
    pot_energy = np.append(pot_energy,U)
    pressure_value = compute_pressure(position, temperature, N)
    pressure = np.append(pressure, pressure_value)
    pressure_cum = np.append(pressure_cum, np.average(pressure))
    specific_heat_value = (2/3-N*(np.average(kinetic_energy**2)/np.average(kinetic_energy)**2-1))**(-1)
    specific_heat = np.append(specific_heat, specific_heat_value)
    pair_corr, bins = corr_function(position, size, delta_r)
    
    #Plot animated parameters
    particles.set_data(position[:,0], position[:,1])
    particles.set_3d_properties(position[:,2])
    Energy_plot.set_data(time, energy)
    T_plot.set_data(time, kinetic_energy) 
    U_plot.set_data(time, pot_energy)
    Pressure_plot.set_data(time, pressure)
    Pressure_cum_plot.set_data(time, pressure_cum)
    Specific_heat_plot.set_data(time, specific_heat)
    Pair_corr_plot.set_data(bins + delta_r/2, pair_corr)
    
    #Error for pressure
    error_pressure = bootstrap(pressure, bootstrap_iterations)
    ax3.errorbar(time, pressure_cum, yerr=error_pressure, c='r')
    error_specific_heat = bootstrap(specific_heat, bootstrap_iterations)
    
    #Update axis during animation
    xmin, xmax = ax2.get_xlim()
    if time[-1]+h >= xmax:
        ax2.set_xlim(xmin, 2*xmax)
        ax2.figure.canvas.draw()
    ymin, ymax = ax2.get_ylim()
    if T >= ymax-2 or U<=ymin+2:
        ax2.set_ylim(ymin*1.2, ymax*1.2)
        ax2.figure.canvas.draw()
    xmin, xmax = ax3.get_xlim()
    if time[-1]+h >= xmax:
        ax3.set_xlim(xmin, 2*xmax)
        ax3.figure.canvas.draw()
    ymin, ymax = ax3.get_ylim()
    if pressure_cum[-1] >= ymax-2:
        ax3.set_ylim(ymin, ymax*1.2)
        ax3.figure.canvas.draw()
    xmin, xmax = ax4.get_xlim()
    if time[-1]+h >= xmax:
        ax4.set_xlim(xmin, 2*xmax)
        ax4.figure.canvas.draw()
 
    return particles, Energy_plot, T_plot, U_plot, Pressure_plot, Pressure_cum_plot, Specific_heat_plot

    
#Animation
anim = animation.FuncAnimation(fig, iteration, interval=10)

plt.show()
