####################################

from fenics import *
from mshr import*
import numpy as np
import sys
#Add the path to Dymms module to the code
sys.path.insert(0, '../../modules/')
from mupopp import *

from scipy.interpolate import griddata
####################################
# Read the nondimensional numbers from
# the .cfg file and
# Initiate the system
####################################

param_file = sys.argv[1]
soliton=compaction(param_file)

####################################
# Output files
####################################
output_dir     = "output/"
extension      = "pvd"   # "xdmf" or "pvd"
initial_porosity_out = File(output_dir + "initial_porosity." + extension,\
                            "compressed")
velocity_out   = File(output_dir + "velocity." + extension, "compressed")
pressure_out   = File(output_dir + "pressure." + extension, "compressed")
porosity_out   = File(output_dir + "porosity." + extension, "compressed")
compact_out   = File(output_dir + "compaction." + extension, "compressed")
gam_out   = File(output_dir + "gamma." + extension, "compressed")

####################################

####################################
# import 3d soliton from Gideon Simpson's code
# (two columns in "input.out" file)
# define origin (max porosity location)
# input.out contains 2 columns of numbers
###################################
def calculateDistance(x1,y1,z1,x2,y2,z2):
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return dist
class Source(Expression):
    def __init__(self,element):
        """Initiates the class"""
        
    def eval(self, value, x):
        #distance from defined origin
        r0 = calculateDistance(x[0],x[1],x[2],x0,y0,z0)
        value[0] = griddata(R_sol, Phi_sol, r0, method = 'linear')

data = np.loadtxt("input.out")                       
R_sol,Phi_sol = 0.04183300132 * data[:,0], 0.1 * data[:,1]
# data is 1D numpy array
x0 = 0.0;
y0 = 0.0;
z0 = -2.0; #Change this value to recenter the soliton vertically

#####################################
# Generate the mesh
#####################################
Lx     =  4.0
Ly     =  4.0
Lz     =  8.0
x0_min = -Lx/2
x0_max =  Lx/2
x1_min = -Ly/2
x1_max =  Ly/2
x2_min = -Lz/2
x2_max =  Lz/2
n0     =  12 #64
n1     =  12 # 64
n2     =  12 #64

p1 = Point(x0_min, x1_min, x2_min)
p2 = Point(x0_max, x1_max, x2_max)
mesh = BoxMesh(p1, p2, n0, n1, n2)



#####################################
# Create boundary conditions. Impose
# periodic bc at the top and the
# bottom
#####################################
# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):
    # Bottom boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool( near(x[2], -Lz/2.) and on_boundary)
    # Map Top boundary (H) to Bottom boundary (G)
    def map(self, x, y):
        if near(x[2], Lz/2):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - Lz
        else:
            y[0] = -1000
            y[1] = -1000
            y[2] = -1000

# Create periodic boundary condition
pbc = PeriodicBoundary()
##################################
# Create Function spaces
#################################
# Velocity
V     = VectorElement("Lagrange", mesh.ufl_cell(), 2)
# Pressure
Q     = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
#Compaction
OMEGA = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# Make a mixed space
W     = dolfin.FunctionSpace(mesh, MixedElement([V,Q,OMEGA])\
                             , constrained_domain=pbc)
# Output fuction space
# Porosity
X     = FunctionSpace(mesh, "CG", 2, constrained_domain=pbc)
# Velocity
Y     = VectorFunctionSpace(mesh,"CG",2, constrained_domain=pbc)
# Now impose no slip condition on the vertical
# boundaries
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
#####################################
# Prescribe No-slip boundary condition for velocity
# along the vertical walls of the domain
#####################################
noslip = Constant((0.0, 0.0, 0.0))


bc0 = DirichletBC(W.sub(0), noslip, boundary_L)
bc1 = DirichletBC(W.sub(0), noslip, boundary_R)
bc2 = DirichletBC(W.sub(0), noslip, boundary_F)
bc3 = DirichletBC(W.sub(0), noslip, boundary_B)

# Collect boundary conditions
bcs = [bc0, bc1,bc2,bc3]
##################################
# Create initial condnition
#################################
phi_init = Source(element=X.ufl_element())
phi0=Function(X)
phi0.interpolate(phi_init)
initial_porosity_out << phi0

########################################
# This section allows for customized
# application for melting/freezing rate,
# buoyancy, or initial melt fraction
########################################
phi=Function(X)
gam=Function(X)
buyoancy=Function(X)
gam.interpolate(Expression("0.0",degree=1))
buyoancy.interpolate(Expression("1.0",degree=1))
phi.interpolate(phi0)
# Time step
dt = Expression("dt", dt=0.0,degree=1)
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
a, L, b = soliton.momentum_conservation(W, phi0, gam,buyoancy)
###########################################
# Now solve the system of equations
# and return the solution in functions
# u,p, and c for velocity, pressure
# and compaction, respectively
# also returns the number of Krylov iterations
# in the variable niter. These are the
# initial values of the variables.
###########################################

u,p,c,niter = soliton.momentum_solver(W,a,L,b,bcs)

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
dt.dt = 0.05;
a_phi, L_phi,bb_phi = soliton.mass_conservation(X, phi0, u, dt, gam,mesh)

tcount = 1
while (t < soliton.T):
    if t + dt.dt > soliton.T:
        dt.dt = soliton.T - t
    # Solve for U_n+1 and phi_n+1
    ########################################
    # First step, solve the mass conservation
    # equation to get the new value of melt
    # fraction phi
    ########################################
    info("**** t = %g: Solve phi and U" % t)
    phi1=soliton.mass_solver(X,a_phi, L_phi,bb_phi)
    ########################################
    # Update phi0 <- phi1
    ########################################
    phi0.assign(phi1)
    phi.interpolate(phi1)
    ########################################
    # Solve for velocity using the new value
    # of phi
    ########################################
    a, L, b = soliton.momentum_conservation(W, phi, gam,buyoancy)
    u,p,c,niter = soliton.momentum_solver(W,a,L,b,bcs)
    
    if tcount % soliton.out_freq == 0:
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
   
