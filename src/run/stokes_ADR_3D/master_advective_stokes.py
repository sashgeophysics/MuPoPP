from fenics import *
from mshr import*
import numpy, scipy, sys, math
#import matplotlib.pyplot as plt
#Add the path to Mupopp module to the code
#sys.path.insert(0, '../../modules/')

from mupopp import *

import time
start_time = time.time()

#####################################################
parameters["std_out_all_processes"]=False
#set_log_level(30) #Critical info only
#####################################################

#############################
##   Define Input Values   ##
#############################

# Parameters for initializing the object (stokes = StokesAdvection)
Da0  = 1.0e-2
Pe0  = 1.0e-1
# beta = alpha0/phi0
cfl0 = 0.1
# Initial CO3 2- in the system
c01_init=Expression("0.0",degree=1)
# Initial Ca in the system
ca_init=Expression("0.05",degree=1)
# Initial velocity
v_init=Expression(("0.0","0.0","0.0"),degree=3)

# Parameters for iteration (aim for 100 image outputs from the simulation: T0/out_freq0 = 100)
# T0 = L/u0 - this should be the time for movement from in to out if a straight line and constant specified velocity - actual time to travel from in to out may be longer due to tortuosity and diffusion dominant flow
# dt0 = unitless timestep to break up T0

# T0 = L/u0 = 1.3e-3/1e-7 = 13,000 seconds (3hrs 37 mins)
# 13000/pi * 10e7 = 0.0004 years (t0=1)
#T0 = 1 = 3hrs 37mins
#T0 = 100 = 15 days
#T0 = 1000 = 150 days
#T0 = 10000 = 1500 days = 4 years 40 days

# T0 = L/u0 = 1.3e-3/1e-2 = 0.13 seconds
# T0 = 1 = 0.13 seconds
# T0 = 1000 = 130 seconds
# T0 = 10000 = 22 minutes
# T0 = 100000 = 3hrs 37 mins

T0 = 10000
dt0 = 100
out_freq0 = 1

######################
##   Create Files   ##
######################

# Output files for visualisation
file_name      = "BFS2_Da_%.1E_Pe_%.1E"%(Da0,Pe0) #Automatically names the output files
output_dir     = file_name + "_output/"
extension      = "pvd"

velocity_out   = File(output_dir + file_name + "_velocity." + extension, "compressed")
pressure_out   = File(output_dir + file_name + "_pressure." + extension, "compressed")
c0_out         = File(output_dir + file_name + "_concentration0." + extension, "compressed")
c1_out         = File(output_dir + file_name + "_concentration1." + extension, "compressed")
c2_out         = File(output_dir + file_name + "_concentration2." + extension, "compressed")

# Output parameters
def output_write(Da,Pe,cfl,fname= output_dir + "/a_parameters.out"):
    """This function saves the output of iterations"""
    file=open(fname,"a")
    file.write("####################################")
    file.write("\n")
    file.write("Da:  %g\n" %Da0)
    file.write("Pe:  %g\n" %Pe0)
    file.write("cfl:  %g\n" %cfl0)
    file.write("####################################")
    file.close

output_write(Da0,Pe0,cfl0)

##################################
##   Create Complex Functions   ##
##################################

#Define a function h such that c(x,0)=f
class BoundaryConcentration(Expression):
    def __init__(self, mesh, element):
        self.mesh = mesh
        self.element=element
    def eval_cell(self, values, x, ufl_cell):
	f = 0.05*(1.0-tanh(0.0/0.2))/2.0
        values[0] = f       
    def value_shape(self):
        return (1,)

###########################################
##   Mesh Generation + Function Spaces   ##
###########################################

# Set file names for importing mesh files in .h5 format
mesh_fname = "BFS2_mesh.xml"
extension = ".h5"

# Read in the .xml mesh converted to .h5 format
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), "hdf5_" + mesh_fname + extension, "r")
hdf.read(mesh, "/mesh", False)

# Read in the boundaries in .h5 format, called as 'boundaries' to allow for specifying 1, 2, 3 as
# inflow, outflow, walls in the below boundary conditions
#boundaries = FacetFunction("size_t", mesh) # Works but has been depricated in 2017.2.0 and replaced as below
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
hdf.read(boundaries, "/boundaries")

# Define ds (2D surface for flux) and n (unit vector normal to a plane)
ds = Measure("ds", domain = mesh, subdomain_data = boundaries)
n = FacetNormal(mesh)

# Create FunctionSpaces for each unknown to be calculated in

# Velocity
V = VectorElement("Lagrange", mesh.ufl_cell(), 3)
# Pressure
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
#Concentration
Qc = FiniteElement("Lagrange",mesh.ufl_cell(), 1)

# Make a mixed FunctionSpace for all 5 unknown to be calculated in together (velocity, pressure, 3x concentrations(carbonate, Ca and precipitated C)
W = dolfin.FunctionSpace(mesh, MixedElement([V,Q,Qc,Qc,Qc]))

# Make a FunctionSpace for initial concentration to operate in
X  = FunctionSpace(mesh,"CG",1)

# Make a FunctionSpace for initial velocity to operate in
Z = VectorFunctionSpace(mesh,"CG",3)

#############################
##   Boundary Conditions   ##
#############################

# Specify the file from which the boundary regions are being imported
# 1 = inflow, 2 = outflow, 3 = walls as per enGrid

#W.sub() substitutes one of the 5 components of the mixed function space
#W.sub(0) uses the first component, V to then define the bc for velocity

bc1 = DirichletBC(W.sub(0), Constant((1.0,0.0,0.0)), boundaries, 1) #from file: 1 = inflow

bc2 = DirichletBC(W.sub(0), Constant((0.0,0.0,0.0)), boundaries, 3) #from file: 3 = walls, 0 velocity = no flow

# Define c0_input as the BoundaryConcentration function for the concentration functionspace, Qc
c0_input = BoundaryConcentration(mesh,element=Qc)

#bc3 = DirichletBC(W.sub(2), c0_input, boundaries, 1) #from file: 1 = inflow
bc3 = DirichletBC(W.sub(2), Constant(0.025), boundaries, 1) #from file: 1 = inflow

# Collect the boundary conditions
bc = [bc1,bc2,bc3]

#########################
## Initial Conditions  ##
#########################

# The initial solution is a function of the mixed function space with all 4 unknowns, W
sol_initial = Function(W)

# Functions for components

temp2 = Function(X) # Define a function in the function space X
temp2.interpolate(c01_init) # Use the predefined expression for c01_init as the function
c01 = temp2 # Call the initial concentration of CO3 2- c01
c02 = temp2 # Call the initial concentration of solid precipitated C c02
temp3 = Function(Z) # Define a function in the function space Z
temp3.interpolate(v_init) # Use the predefined expression for v_init as the function
vel = temp3 # Call the initial velocity v_init
temp4 = Function(X) # Define a function in the function space X
temp4.interpolate(ca_init) # Use the predefined expression for ca_init as the function
c0ca = temp4 # Call the initial concentration of Ca c0ca
# Assign the initial conditions to the correct parts of the mixed function space
# .sub(2) inserts initial concentration (3rd value in W)
# .sub(3) inserts initial concentration of C (4th value in W)
# .sub(4) inserts initial concentration of C (5th value in W)
# .sub(0) inserts initial velocity (1st value in W)
assign(sol_initial.sub(2), c01)
assign(sol_initial.sub(3), c0ca)
assign(sol_initial.sub(4), c02)
assign(sol_initial.sub(0), vel)

###############
##  Solving  ##
###############

# Create the darcy object from the mupopp module using the predefined values at the top
stokes = StokesAdvection(Da=Da0,Pe=Pe0,cfl=cfl0)

# The solution, sol is a function of the mixed function space for all 5 unknowns, W
sol=Function(W)

# Define the iteration parameters from the top
T = T0
dt = dt0
t = dt

# Empty arrays to be filled with flux values and time values at each time step
flux=np.array([])
flux3=np.array([])
mass=np.array([])
time_array=np.array([])
avg_conc=np.array([])

one = Expression(("1.0"),degree=1)
temp5 = Function(X)
temp5.interpolate(one)

# Iteration number and definition of output frequency for a result from the top
i = 1
out_freq = out_freq0

# So long as the time step - total time is less than 0 perform the calculation
# Every iteration adds dt0 to t, so when t gets to be equal to the total time specified, T it will stop
# dt0 = 0.01
# Initially t = dt so t = 0.01, then each timestep: t=0.02, t=0.03...
while t - T < DOLFIN_EPS:
    # We solve for a and L defined in the module by the d_a_r_p_r function
    # We provide the mixed function space, mesh, time step size, source term, initial conditions and flow direction
    a,L=stokes.stokes_advection_diffusion_three_component(W,mesh,sol_prev=sol_initial,dt=dt0,K=0.1,zh=Constant((1.0,0.0,0.0)))
    # Solve for a and L to get the solution, sol which contains all 5 values for the given boundary conditions
    #solve(a==L,sol,bc)
    solve(a==L,sol,bc,solver_parameters={'linear_solver':'mumps'})
    # Give the solution to the function of the shared function space 
    sol_initial = sol
    # Split the solution into the 4 parts of the mixed function space: velcoity, pressure, conc0, conc1
    u,p,c00,c01,c02 = sol.split()
    # Write the results to the files and rename the result. a = CO3 2-, b = Ca, c = solid precipitated C
    if i % out_freq == 0:
	u.rename("Velocity","")
	velocity_out << u
	p.rename("Pressure","")
	pressure_out << p
        c00.rename("[a]","")
        c0_out << c00
        c01.rename("[b]","")
        c1_out << c01
	c02.rename("[C]","")
        c2_out << c02
	# Calculate CO3 2- input flux
        flux1 = assemble(c00*dot(u, -n)*ds(1))
        flux = np.append(flux,flux1)
        time_array=np.append(time_array,t)
	# Calculate C mass precipitated in the domain
        mass1 = assemble(c02*dx)
        mass = np.append(mass,mass1)
	avg_conc1 = assemble(c02*dx)/assemble(temp5*dx)
	avg_conc = np.append(avg_conc,avg_conc1)
    # Move to next interval and adjust boundary condition
    info("time t =%g\n" %t)
    info("iteration =%g\n" %i)
    #print 'iteration',i
    t += dt
    i += 1
    flux_file_CO3=output_dir + file_name + "_CO3_flux.csv"
    np.savetxt(flux_file_CO3,(time_array,flux),delimiter=',')
    mass_file_C=output_dir + file_name + "_C_mass.csv"
    np.savetxt(mass_file_C,(time_array,mass,avg_conc),delimiter=',')
    info("Time elapsed %s seconds ---" % (time.time() - start_time))

info("Finished execution in %s seconds ---" % (time.time() - start_time))
