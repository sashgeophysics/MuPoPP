### this script compares the numerical solution to the 2D wedge
### to the analytical solution from Spiegelman and McKenzie(1987)
from dolfin import *
import numpy as np
import scipy, sys, math
import matplotlib.pyplot as plt
#Add the path to Dymms module to the code
sys.path.insert(0, '../../modules/')
from dymms import*
import core
from analytical import*

## Create an object for wedge flow and generate a mesh
benchmark_flow=wedge_flow(phi0=0.1,u0=1.0)
mesh=wedge_mesh(n=4) #higher value of n will give a refined mesh
#Calculate the analytical solutions for velocity and pressure
vel_analytical,p_analytical=benchmark_flow.calculate_wedge_flow(mesh)
fname="output/analytical_pressure.pvd"
C_out=File(fname)
C_out <<p_analytical
fname="output/analytical_velocity.pvd"
v_out=File(fname)
v_out<<vel_analytical
##############################################
## Now calculate the numerical solution
## This part is similar to the subduction code
###############################################
####################################################
#Set parameters for petsc
#####################################################
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Test for PETSc or Epetra
if not has_linear_algebra_backend("PETSc") and not \
   has_linear_algebra_backend("Epetra"):
    info("DOLFIN has not been configured with Trilinos \
    or PETSc. Exiting.")
    exit()

if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this demo is only available when DOLFIN is\
    compiled with AMG "
         "preconditioner, Hypre or ML.")
    exit()

if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    info("Default linear algebra backend was not compiled \
    with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()
#####################################
# import nondimensional parameters
####################################
    
param_file="../subduction_2D/subduction.cfg"
param   = core.parse_param_file(param_file)

# General parameters
logname  = param['logfile']
out_freq = param['out_freq']

#Set the problem domain
inflow=domain()

# Set time stepping parameters
T = param['T']
dt = param['dt']
cfl = param['cfl']

#Set nondimensional paramters
inflow.da=param['da']
inflow.R=param['R']
inflow.B=param['B']
inflow.theta=param['theta']
inflow.dL=param['dL']
# Output files for quick visualisation
output_dir     = "output/"
extension      = "pvd"  
initial_porosity_out = File(output_dir + "initial_porosity."\
                            + extension, "compressed")
velocity_out   = File(output_dir + "velocity." + extension, "compressed")
pressure_out   = File(output_dir + "pressure." + extension, "compressed")
porosity_out   = File(output_dir + "porosity." + extension, "compressed")
compact_out   = File(output_dir + "compaction." + extension, "compressed")
gam_out   = File(output_dir + "gamma." + extension, "compressed")
gam_out   = File(output_dir + "gamma." + extension)
cont_out   = File(output_dir + "contiguity." + extension)
vs_out   = File(output_dir + "vs." + extension)
shear_out   = File(output_dir + "shear." + extension)
######################################################################
# Function Spaces
######################################################################

# Define function spaces
# Velocity
V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
# Pressure
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
#Compaction
OMEGA = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# Make a mixed space
W=dolfin.FunctionSpace(mesh, MixedElement([V,Q,OMEGA]))

# Output fuction space
# Porosity
X = FunctionSpace(mesh, "CG", 1)
# Velocity
Y = VectorFunctionSpace(mesh, "CG", 2)
# Pressure and porosity
Z = FunctionSpace(mesh, "CG", 1)

######################################################################
# Boundaries
######################################################################
# Define boundary functions
def slant_boundary(x, on_boundary):
    return on_boundary and x[0] <1.1- DOLFIN_EPS and x[1]<1.1-DOLFIN_EPS
def top(x, on_boundary):
    return x[1] > (1.0- DOLFIN_EPS)
def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
def bottom(x, on_boundary):
    return x[1] < (DOLFIN_EPS)

# Magnitude of subduction velocity
u0=1.0

bc1 = DirichletBC(W.sub(0),(u0,-u0) , slant_boundary)
bc2 = DirichletBC(W.sub(0), (0.0,0.0), top)
bcs=[bc1,bc2]


# ======================================================================
# Solution functions
# ======================================================================


# Split-field solution

U = Function(W)
u, p, c = U.split()


# Porosity at time t_n
phi0 = Function(X)

# Porosity at time t_n+1
phi1 = Function(X)

#Melting rate

gam=Function(X)

# Spatial buoyancy
buyoancy=Function(X)
# ======================================================================
#  Weak formulations
# ======================================================================
a_phi, L_phi,bb_phi = inflow.mass_conservation(X, phi0, u, dt, gam,mesh)
a_stokes, L_stokes, b = inflow.momentum_conservation(W, phi0, gam,\
                                                     buyoancy)
######################################################################
# Initial porosity condition
######################################################################
phi_init=Expression("0.1",degree=1)
#phi_init = Source()

phi0.interpolate(phi_init)
initial_porosity_out << phi0

phi00=0.1
#Function describing the rate of melting
# (-ve for freezing) as a function of space
gam_temp=Expression("0.0",degree=1)
gam.interpolate(gam_temp)
gam_out << gam

rho_temp=Expression("0.0",degree=1)
buyoancy.interpolate(rho_temp)


# Porosity
phi = Function(Z)
phi.interpolate(phi0)

t = 0.0
# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

######################################################################
#  Initial velocity field
######################################################################
# Create Krylov solver and AMG preconditioner
solver = KrylovSolver(krylov_method, "amg")
solver.parameters["relative_tolerance"] = 0.000001
solver.parameters["maximum_iterations"] = 3000 
solver.parameters["monitor_convergence"] = True

# Assemble system
A_stokes, b_stokes = assemble_system(a_stokes, L_stokes, bcs)

# Assemble preconditioner system
P, btmp = assemble_system(b, L_stokes, bcs)

# Associate operator (A) and preconditioner matrix (P)
solver.set_operators(A_stokes, P)

# Solve
solver.solve(U.vector(), b_stokes)

# Get sub-functions
u, p, c = U.split()

##################################
## Out put to files
##################################
# Velocity field
u.rename("velocity", "")
velocity_out  << u

# Pressure field
p.rename("pressure", "")
pressure_out  << p

# Compaction field
c.rename("compaction", "")
compact_out  << c

# Porosity field
phi.rename("porosity", "")
porosity_out  << phi
