###################################################
## Copyright Saswata Hier-Majumder, September 2017
## This program creates a sphere
## melt distribution 
## and solves for the velocity, pressure, and
## and compaction for Dirichlet BC
## Rewritten for MuPopp, July, 2018
#################################################### 

####################################################

from fenics import *
from mshr import*
import numpy, scipy, sys, math
#Add the path to Dymms module to the code
sys.path.insert(0, '../../modules/')
from mupopp import *

#####################################################

param_file = sys.argv[1]
sphere=compaction(param_file)
#####################################################
# Output files for quick visualisation
output_dir     = "output/"
extension      = "pvd"   # "xdmf" or "pvd"
initial_porosity_out = File(output_dir + "initial_porosity." + extension,\
                            "compressed")
velocity_out   = File(output_dir + "velocity." + extension, "compressed")
pressure_out   = File(output_dir + "pressure." + extension, "compressed")
porosity_out   = File(output_dir + "porosity." + extension, "compressed")
compact_out   = File(output_dir + "compaction." + extension, "compressed")
gam_out   = File(output_dir + "gamma." + extension, "compressed")

######################################################################
# Create a sphere Mesh 
######################################################################
s1 = Sphere(Point(0, 0, 0), 1.0)
mesh = generate_mesh(s1, 10)
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
class sphere_surface(SubDomain):
    def inside(self,x,on_boundary):
        r = sqrt(x[0]**2 + x[1]**2+x[2]**2)
        return on_boundary and near(r, 1.0,0.05)
surf=sphere_surface()


# Prescribe No-slip boundary condition for velocity
# on the sphere surface

noslip = Constant((0.0, 0.0, 0.0))
straining_flow=Expression(("0.05*x[0]","0.05*x[1]","-0.1*x[2]"),degree=2)
bcs = DirichletBC(W.sub(0),straining_flow, surf)

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
# Formulations


###########################################
# Initial  condition and known functions
###########################################
# Create an initial melt distribution
phi0.interpolate(Expression("0.01+0.001*x[0]*x[1]",degree=2))
initial_porosity_out << phi0
# Porosity
phi = Function(Z)
phi.interpolate(phi0)

t = 0.0
# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;
###########################################
# Get the bilinear form for the
# momentum conservation equation
# a = L
# and the preconditioner for the moemntum
# solver b
###########################################
a, L, b = sphere.momentum_conservation(W, phi0, gam,buyoancy)
###########################################
# Now solve the system of equations
# and return the solution in functions
# u,p, and c for velocity, pressure
# and compaction, respectively
# also returns the number of Krylov iterations
# in the variable niter. These are the
# initial values of the variables.
###########################################

u,p,c,niter = sphere.momentum_solver(W,a,L,b,bcs)

# Write data to files for  visualisation
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
dt.dt = 0.1;
a_phi, L_phi,bb_phi = sphere.mass_conservation(X, phi0, u, dt, gam,mesh)
# Solver matrices
A_phi, A_stokes = Matrix(), Matrix()

# Solver RHS
b_phi, b_stokes = Vector(), Vector()





tcount = 1
while (t < sphere.T):
    if t + dt.dt > sphere.T:
        dt.dt = sphere.T - t



    # Solve for U_n+1 and phi_n+1
    ########################################
    # First step, solve the mass conservation
    # equation to get the new value of melt
    # fraction phi
    
    ########################################
    info("**** t = %g: Solve phi and U" % t)
    phi1=sphere.mass_solver(X,a_phi, L_phi,bb_phi)
    ########################################
    # Update phi0 <- phi1
    ########################################
    phi0.assign(phi1)
    phi.interpolate(phi1)
    ########################################
    # Solve for velocity using the new value
    # of phi
    ########################################
    a, L, b = sphere.momentum_conservation(W, phi, gam,buyoancy)
    u,p,c,niter = sphere.momentum_solver(W,a,L,b,bcs)
    
    if tcount % sphere.out_freq == 0:
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




