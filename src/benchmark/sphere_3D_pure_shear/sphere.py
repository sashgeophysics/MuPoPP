####################################
## This file is a part of MuPoPP 
## This file calculates the solution
## for compaction equations for one
## time step. Compares the numerical
## solution with analytical solution
## from
## Hier-Majumder, S. (2018). Analytical solution
## for two-phase flow within and outside a sphere under pure shear.
## Journal of Fluid Mechanics, 948
## doi:10.1017/jfm.2018.373
##
## The boundary conditions
## are given by
## u = E.r on Gamma
## grad(p).n = 0 on Gamma
## Under this condition, the analytical 
## solution for a sphere is given by
## u = E.r in Omega
## p = 0   in Omega
## C = 0   in Omega
## Section 3.5.1 of Hier-Majumder 2018.
## Copyright, Saswata Hier-Majumder
## July 6th, 2018
####################################

from fenics import *
from mshr import*
import numpy as np
import sys
#Add the path to Dymms module to the code
sys.path.insert(0, '../../modules/')
from mupopp import *

####################################
# Read the nondimensional numbers from
# the .cfg file and
# Initiate the system
####################################

param_file = sys.argv[1]
sphere=compaction(param_file)

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

###################################
## Create a simple sphere  mesh
##################################
s1 = Sphere(Point(0, 0, 0), 1.0)
mesh = generate_mesh(s1, 12)

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
W     = dolfin.FunctionSpace(mesh, MixedElement([V,Q,OMEGA]))
# Output fuction space
# Porosity
X     = FunctionSpace(mesh, "CG", 2)
# Velocity
Y     = VectorFunctionSpace(mesh,"CG",2)

########################################
# Create initial and boundary conditions
########################################
phi_init = Expression("0.01",degree=2)
phi0=Function(X)
phi0.interpolate(phi_init)
initial_porosity_out << phi0
# Create a class that returns the sphere surface
# of radius 1
class sphere_surface(SubDomain):
    def inside(self,x,on_boundary):
        r = sqrt(x[0]**2 + x[1]**2+x[2]**2)
        return on_boundary and near(r, 1.0,0.05)
#Instantiate the class into an object surf    
surf=sphere_surface()
#Create an expression for velocity on the surface of
# of the sphere
straining_flow=Expression(("0.05*x[0]","0.05*x[1]","-0.1*x[2]"),degree=2)
#Assign the expression for velocity as the Dirichlet BC
bcs = DirichletBC(W.sub(0),straining_flow, surf)
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
phi.interpolate(Expression("0.01",degree=2))

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

###########################################
### Compare results with analytical solution
############################################
uh=Function(Y)
ph=Function(X)
ch=Function(X)
uh.interpolate(straining_flow)
ph.interpolate(Expression('0.0',degree=2))
ch.interpolate(Expression('0.0',degree=2))

###########################
## output iteration details
############################

h = mesh.hmax()
#Print error norm of velocity
l2_u=errornorm(u,uh,'L2')
l2_p=errornorm(p,ph,'L2')
l2_c=errornorm(c,ch,'L2')

iteration_output_info(niter,h,l2_u,l2_p,l2_c)
iteration_output_write(niter,h,l2_u,l2_p,l2_c)
