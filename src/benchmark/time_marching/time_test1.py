###################################################
## This is a part of Mupopp
## Copyright Saswata Hier-Majumder, July, 2016
## Modified on July 2018
## This program creates a 3D cube with an initial
## melt distribution 
## and solves for the velocity, pressure, and
## and compaction for Dirichlet BC
####################################################

from fenics import *
from mshr import*
import numpy, scipy, sys, math
#Add the path to Dymms module to the code
sys.path.insert(0, '../../modules/')
from mupopp import *

#####################################################



#####################################################################
## Read the valu of parameters from the input file
## and create an object using those parameters
#####################################################################
param_file = sys.argv[1]
test1=compaction(param_file)
#####################################################################
# Output files for quick visualisation
#####################################################################
output_dir     = "output/"
extension      = "pvd"   # "xdmf" or "pvd"
initial_porosity_out = File(output_dir + "initial_porosity." + extension,\
                            "compressed")
velocity_out   = File(output_dir + "velocity." + extension, "compressed")
pressure_out   = File(output_dir + "pressure." + extension, "compressed")
porosity_out   = File(output_dir + "porosity." + extension, "compressed")
compact_out    = File(output_dir + "compaction." + extension, "compressed")
gam_out        = File(output_dir + "gamma." + extension, "compressed")

######################################################################
# Mesh and Paramters
######################################################################
# Lengths of the cube
Lx     =  4.0
Ly     =  4.0
Lz     =  8.0
x0_min = -Lx/2
x0_max =  Lx/2
x1_min = -Ly/2
x1_max =  Ly/2
x2_min =  0.0
x2_max =  Lz
# Mesh density in the three directions
# Change the numberfor a higher mesh density
n0     =  6 
n1     =  6 
n2     =  12


# This following bit is a flag for working between different versions
# of Dolfin. 
flg = dolfin.dolfin_version() 
flg = float(flg[0:3])

p1 = Point(x0_min, x1_min, x2_min)
p2 = Point(x0_max, x1_max, x2_max)

if flg < 1.6:
    mesh = BoxMesh(x0_min, x1_min, x2_min, x0_max, x1_max, x2_max, n0, n1, n2)
else:
    mesh = BoxMesh(p1, p2, n0, n1, n2)

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

def boundary_L(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[0], x0_min,DOLFIN_EPS)
def boundary_R(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[0], x0_max,DOLFIN_EPS)
def boundary_F(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[1], x1_max,DOLFIN_EPS)
def boundary_B(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[1], x1_min,DOLFIN_EPS)

# Prescribe No-slip boundary condition for velocity
# along the vertical walls of the domain
noslip = Constant((0.0, 0.0, 0.0))


bc0 = DirichletBC(W.sub(0), noslip, boundary_L)
bc1 = DirichletBC(W.sub(0), noslip, boundary_R)
bc2 = DirichletBC(W.sub(0), noslip, boundary_F)
bc3 = DirichletBC(W.sub(0), noslip, boundary_B)

zero = Constant(0)
# Collect boundary conditions
bcs = [bc0, bc1,bc2,bc3]

# ======================================================================
# Solution functions
# ======================================================================
# Porosity at time t_n
phi0 = Function(X)
# Porosity at time t_n+1
phi1 = Function(X)

#Melting rate
gam=Function(X)
gam.interpolate(Expression("0.0",degree=1))
#Density 
# Spatial buoyancy
buyoancy=Function(X)
buyoancy.interpolate(Expression("1.0",degree=1))

# Time step
dt = Expression("dt", dt=0.0,degree=1)
# ======================================================================
#  Develop the weak formulation
# ======================================================================
# Time step
dt = Expression("dt", dt=0.0,degree=1)

######################################################################
# Initial  condition and known functions
######################################################################
# Create an initial melt distribution
phi_init = Expression("0.05*(exp(-x[0]*x[0]-x[1]*x[1]-(x[2]-0.5)*(x[2]-0.5))/0.5)+0.1",degree=2)
phi0.interpolate(phi_init)
initial_porosity_out << phi0

#Function describing the rate of melting
# (-ve for freezing) as a function of space
gam.interpolate(Expression("0",degree=1))
gam_out << gam
# Buoyancy
buyoancy.interpolate(Expression("1.0",degree=1))
# Porosity
phi = Function(Z)
phi.interpolate(phi0)

t = 0.0
# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;
######################################################################
# Get the bilinear form for the
# momentum conservation equation
# a = L
# and the preconditioner for the moemntum
# solver b
######################################################################
a, L, b = test1.momentum_conservation(W, phi0, gam,buyoancy)
###########################################
# Now solve the system of equations
# and return the solution in functions
# u,p, and c for velocity, pressure
# and compaction, respectively
# also returns the number of Krylov iterations
# in the variable niter. These are the
# initial values of the variables.
###########################################

u,p,c,niter = test1.momentum_solver(W,a,L,b,bcs)

# Write data to files for  visualization
# Velocity field
u.rename("velocity", "")
velocity_out  << u

# Pressure field
p.rename("pressure", "")
pressure_out  << p

# Compaction field
c.rename("compaction", "")
compact_out  << c
# ======================================================================
#  Time loop
# ======================================================================

# Set up time setp
dt.dt = test1.dt;
#a_phi, L_phi,bb_phi = test1.mass_conservation(X, phi0, u, dt, gam,mesh)

tcount = 1
while (t < test1.T):
    if t + dt.dt > test1.T:
        dt.dt = test1.T - t
    # Solve for U_n+1 and phi_n+1
    ########################################
    # First step, solve the mass conservation
    # equation to get the new value of melt
    # fraction phi
    ########################################
    info("**** t = %g: Solve phi and U" % t)
    #phi1=test1.mass_solver(X,a_phi, L_phi,bb_phi)
    phi_temp = Expression("(0.15*(exp(-x[0]*x[0]-x[1]*x[1]-(x[2]-0.5*time)\
    *(x[2]-0.5*time))/0.5)+0.1)",degree=2,time=t)
    phi1.interpolate(phi_temp)
    ########################################
    # Update phi0 <- phi1
    ########################################
    phi0.assign(phi1)
    phi.interpolate(phi1)
    ########################################
    # Solve for velocity using the new value
    # of phi
    ########################################
    a, L, b = test1.momentum_conservation(W, phi, gam,buyoancy)
    u,p,c,niter = test1.momentum_solver(W,a,L,b,bcs)
    
    if tcount % test1.out_freq == 0:
        # Write data to files for quick visualisation
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
        
        
        info("output results")
    info("**** New time step dt = %g\n" % dt.dt)
    t      += dt.dt
    tcount += 1
    info("**** New time t = %g\n" % t)
   
