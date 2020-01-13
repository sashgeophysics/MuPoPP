from fenics import *
from mshr import*
import numpy, scipy, sys, math
#import matplotlib.pyplot as plt
#Add the path to Mupopp module to the code
#sys.path.insert(0, '../../modules/')

from mupopp import *

# Add a reference start time for time keeping
import time
start_time = time.time()

# Reduce command line output to 1 processor
parameters["std_out_all_processes"]=False

#############################
##   Define Input Values   ##
#############################

# Parameters for initialising the object of the class (stokes = StokesAdvection)
Da0  = 1.0e-1
Pe0  = 1.0e-1
# Initial CO3 2- mass fraction in the system
c01_init=Expression("0.0",degree=1)
# Initial anorthite (Ca) mass fraction in the system
ca_init=Expression("0.10",degree=1)
anorthite = 0.10
# Initial velocity
v_init=Expression(("0.0","0.0","0.0"),degree=3)

# Parameters for iteration (aim for 100 image outputs from the simulation: T0/out_freq0 = 100)
# Non dimensional time, T0 can be calculated using: T0 = L/u0 (L=characteristic length, u0 = velocity)
# T0 is calculated in seconds, this can then be used to work out how long to run the simulation as:

# T0 = 1e-3/1e-7 = 10,000 seconds 
# T0 = 1 = 2 hrs 46 mins 40 secs
# T0 = 10 = 1 day 3 hrs 46 mins 40 secs
# T0 = 100 = 11 days 13 hrs 46 mins 40 secs
# T0 = 1,000 = 115 days 17 hrs 46 mins 40 secs
# T0 = 10,000 = 3 years 62 days 9 hrs 46 mins 40 secs
# T0 = 100,000 = 31 years 259 days 1 hr 46 mins 40 secs

# To work out years from seconds you can also do:
# seconds/pi * 10e7

# Total simulation time
T0 = 100000
# Time step size
dt0 = 100
# Every n timesteps data will be output
out_freq0 = 1

######################
##   Create Files   ##
######################

# Output files for visualisation
file_name      = "An_%3.2f_Da_%.1E_Pe_%.1E"%(anorthite,Da0,Pe0)
output_dir     = file_name + "_output/"
extension      = "pvd"

velocity_out   = File(output_dir + file_name + "_velocity." + extension, "compressed")
pressure_out   = File(output_dir + file_name + "_pressure." + extension, "compressed")
c0_out         = File(output_dir + file_name + "_concentration0." + extension, "compressed")
c1_out         = File(output_dir + file_name + "_concentration1." + extension, "compressed")
c2_out         = File(output_dir + file_name + "_concentration2." + extension, "compressed")

###########################################
##   Mesh Generation + Function Spaces   ##
###########################################

# Set file names for importing mesh files in .h5 format
mesh_fname = "hdf5_sample_mesh.xml"
extension = ".h5"

# Read in the .xml mesh converted to .h5 format
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), "hdf5_" + mesh_fname + extension, "r")
hdf.read(mesh, "/mesh", False)

# Read in the boundaries, called as 'boundaries' to allow for specifying 1, 2, 3 as
# inflow, outflow, walls in the below boundary conditions
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
hdf.read(boundaries, "/boundaries")

# Define ds (2D surface for flux) and n (unit vector normal to a plane) for calculated input carbon flux
ds = Measure("ds", domain = mesh, subdomain_data = boundaries)
n = FacetNormal(mesh)

# Create FunctionSpaces for each unknown to be calculated in

# Velocity
V = VectorElement("Lagrange", mesh.ufl_cell(), 3)
# Pressure
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# Concentrations
Qc = FiniteElement("Lagrange",mesh.ufl_cell(), 1)

# Make a mixed FunctionSpace for all 5 unknowns to be calculated in together
# (velocity, pressure, 3x concentrations (carbonic acid, anorthite and precipitated C)
W = dolfin.FunctionSpace(mesh, MixedElement([V,Q,Qc,Qc,Qc]))

# Make a FunctionSpace for initial concentrations to operate in
X = FunctionSpace(mesh,"CG",1)

# Make a FunctionSpace for initial velocity to operate in
Z = VectorFunctionSpace(mesh,"CG",3)

#############################
##   Boundary Conditions   ##
#############################

# 1 = inflow, 2 = outflow, 3 = walls as labelled in enGrid

#W.sub() substitutes one of the 5 components of the mixed function space
#W.sub(0) uses the first component, V to then define the bc for velocity

bc1 = DirichletBC(W.sub(0), Constant((1.0,0.0,0.0)), boundaries, 1) #from file: 1 = inflow
# Velocity is defined using Pe as we calculate this based on dimensional velocity
# therefore we set velocity to 1.0 here to retain the values 'stored' within Pe

bc2 = DirichletBC(W.sub(0), Constant((0.0,0.0,0.0)), boundaries, 3) #from file: 3 = walls
# 0 velocity on the walls imposes no-flow conditions

bc3 = DirichletBC(W.sub(2), Constant(0.1), boundaries, 1) #from file: 1 = inflow
# 0.1 (10%) is the mass fraction of the total system which is carbonic acid entering from inflow

# Collect the boundary conditions
bc = [bc1,bc2,bc3]

#########################
## Initial Conditions  ##
#########################

# The initial solution is a function of the mixed function space with all 5 unknowns, W
sol_initial = Function(W)

# Functions for components

# Define a function in the function space X and use the 
# predefined expression for c01_init as the function
temp1 = Function(X) 
temp1.interpolate(c01_init)
# As there is the same initial amount of carbonic acid (c01) and precipitated C (c02)
# in the system we use the c01_init function and assign it to both
c01 = temp1
c02 = temp1

# Define a function in the function space Z and use the 
# predefined expression for v_init as the function
temp2 = Function(Z)
temp2.interpolate(v_init)
# Call the initial velocity, vel and assign it the value of v_init
vel = temp2

# Define a function in the function space X and use the 
# predefined expression for ca_init as the function
temp3 = Function(X)
temp3.interpolate(ca_init)
# Call the initial concentration of anorthite, c0ca and assign it the value of ca_init
c0ca = temp3

# Assign the initial conditions to the correct parts of the mixed function space 
# of the initial solution, sol_initial
# .sub(x) inserts the following argument into the relevant slot of sol_initial
# which uses the W function space so 0 = velocity, 1 = pressure, 2 = [carbonic acid] etc...
assign(sol_initial.sub(0), vel)
assign(sol_initial.sub(2), c01)
assign(sol_initial.sub(3), c0ca)
assign(sol_initial.sub(4), c02)

###############
##  Solving  ##
###############

# Create the stokes object from the mupopp module using the predefined values at the top
stokes = StokesAdvection(Da=Da0,Pe=Pe0)

# The solution, sol is a function of the mixed function space, W for all 5 unknowns
sol=Function(W)

# Define the iteration parameters from the top
T = T0
dt = dt0
t = dt

# Empty arrays to be filled with flux values and time values at each time step
iflux=np.array([])
volint_conc=np.array([])
rate_conc=np.array([])
time_array=np.array([])
avg_conc=np.array([])
volume=np.array([])
oflux=np.array([])

one = Expression(("1.0"),degree=1)
temp4 = Function(X)
temp4.interpolate(one)

# Iteration number and definition of output frequency for a result from the top
i = 1
out_freq = out_freq0

# So long as the time step - total time is less than 0 perform the calculation
# Every iteration adds dt0 to t, so when t gets to be equal to the total time specified,
# T it will stop. If dt0 = 0.01 initially t = dt so t = 0.01, then each timestep: t=0.02, t=0.03...
while t - T < DOLFIN_EPS:
    # We solve for a and L defined as the linear and bilinear part of F in the module function
    # We provide the mixed function space, mesh, initial conditions, time step size and flow direction
    a,L=stokes.stokes_ADR_precipitation(W,mesh,sol_prev=sol_initial,dt=dt0,zh=Constant((1.0,0.0,0.0)))
    # Solve for a and L to get the solution, sol which contains all 5 values for the given boundary conditions
    solve(a==L,sol,bc,solver_parameters={'linear_solver':'mumps'}) # MUMPS allows for larger memory usage
    # Set the initial conditions for the next time step as the solution to the previous step 
    sol_initial = sol
    # Split the solution into the 5 parts of the mixed function space:
    # velocity, pressure, [carbonic acid], [anorthite], [carbonate precipitate]
    u,p,c00,c01,c02 = sol.split()
    # Write the results to the files and rename the results
    if i % out_freq == 0:
	u.rename("Velocity","")
	velocity_out << u
	p.rename("Pressure","")
	pressure_out << p
        c00.rename("[H2CO3]","")
        c0_out << c00
        c01.rename("[Anorthite]","")
        c1_out << c01
	c02.rename("[CaCO3]","")
        c2_out << c02

	# Calculate H2CO3 concentration input flux, integrated over the inflow from file: 1
        iflux1 = assemble(c00*dot(u, -n)*ds(1))
        iflux = np.append(iflux,iflux1) # Store the input flux at each step, iflux1 in the empty array, iflux
        

	time_array=np.append(time_array,t) # Store the total time at each time step, t in the empty array, time_array
	
	# Calculate H2CO3 concentration output flux, integrated over the outflow from file: 2
        oflux1 = assemble(c00*dot(u, -n)*ds(2))
        oflux = np.append(oflux,oflux1) # Store the outflow flux at each step, oflux1 in the empty array, oflux

	# Calculate C concentration precipitated in the whole domain, dx during the given time step only (not a cummulative amount)
        volint_conc1 = assemble(c02*dx)
        volint_conc = np.append(volint_conc,volint_conc1) # Store the total precipitated concentration for a given step, volint_conc1 in the empty array, volint_conc

	# Calculate the production rate of CaCO3 concentration for each given time step
	Total_molar_mass = 698.0
        An_frac = 278.0/Total_molar_mass
        H2CO3_frac = 62.0/Total_molar_mass
        CaCO3_frac = 100.0/Total_molar_mass	
	rate_conc1 = assemble(CaCO3_frac*Da0*c00*c01*dx)
	rate_conc = np.append(rate_conc,rate_conc1)	

   	# Calculate the average concentration of precipitate across the whole domain, dx
	avg_conc1 = assemble(c02*dx)/assemble(temp4*dx)
	avg_conc = np.append(avg_conc,avg_conc1) # Store the average concentration precipitated at each step, avg_conc1 in the empty array, avg_conc 
	
	# Calculate the volume of the domain - will be constant
	volume1 = assemble(temp4*dx)
	volume = np.append(volume,volume1)

    # Move to the next time step
    info("time t =%g\n" %t)
    info("iteration =%g\n" %i)
    t += dt
    i += 1
    # Write the flux and concentration data out:
    mass_file=output_dir + "0_non_dimensional_outputs.csv"
    np.savetxt(mass_file,(time_array,volume,iflux,oflux,volint_conc,rate_conc,avg_conc),delimiter=',')
    
    # Print out the total elapsed time after each time step
    info("Time elapsed %s seconds ---" % (time.time() - start_time))

# Print total simulation time once complete
info("Finished execution in %s seconds ---" % (time.time() - start_time))
