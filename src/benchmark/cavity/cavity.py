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
cavity=compaction(param_file)

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
## Create a simple cavity  mesh
##################################
# Lengths of the cube
Lx     =  16.0
Ly     =  8.0
Lz     =  8.0
x0_min = -Lx/2
x0_max =  Lx/2
x1_min = -Ly/2
x1_max =  Ly/2
x2_min = -Lz/2
x2_max =  Lz/2

#s1 = Sphere(Point(0, 0, 0), 1.0)
#b1 = Box(Point(x0_min,x1_min,x2_min), Point(x0_max,x1_max,x2_max))
#mesh = generate_mesh(b1-s1, 15)

c1=Circle(Point(0.0,0.0),1.0)
s1=Rectangle(Point(x0_min,x1_min),Point(x0_max,x1_max))
mesh = generate_mesh(s1-c1,25)
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
# Create a class that returns the cavity surface
# of radius 1
class cavity_surface(SubDomain):
    def inside(self,x,on_boundary):
        r = sqrt(x[0]**2 + x[1]**2+x[2]**2)
        return on_boundary and near(r, 1.0,0.05)
class cavity_surface_2D(SubDomain):
    def inside(self,x,on_boundary):
        r = sqrt(x[0]**2 + x[1]**2)
        return on_boundary and near(r, 1.0,0.05)
#Instantiate the class into an object surf    
surf=cavity_surface_2D()
#Create an expression for velocity on the surface of
# of the cavity
straining_flow=Expression(("0.05*x[0]","0.05*x[1]","-0.1*x[2]"),degree=2)
straining_flow_2D=Expression(("0.05*x[1]","-0.0*x[1]"),degree=2)
noslip_2D = Constant((0.0, 0.0))
zero= Constant(0.0)
def boundary_L(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[0], x0_min,DOLFIN_EPS)
def boundary_R(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[0], x0_max,DOLFIN_EPS)
def boundary_Top(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[1], x1_max,DOLFIN_EPS)
def boundary_Bot(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[1], x1_min,DOLFIN_EPS)
def boundary_F(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[2], x2_max,DOLFIN_EPS)
def boundary_B(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[1], x2_min,DOLFIN_EPS)

#Assign the expression for velocity as the Dirichlet BC
bc0 = DirichletBC(W.sub(0), straining_flow_2D, boundary_L)
bc1 = DirichletBC(W.sub(0), straining_flow_2D, boundary_R)
bc2 = DirichletBC(W.sub(0), straining_flow_2D, boundary_Top)
bc3 = DirichletBC(W.sub(0), straining_flow_2D, boundary_Bot)
bc4 = DirichletBC(W.sub(0), noslip_2D, surf)
bcs = [bc0,bc1,bc2,bc3,bc4]
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
a, L, b = cavity.momentum_conservation(W, phi0, gam,\
                                       buyoancy,zhat=Constant((0.0,1.0)))

###########################################
# Now solve the system of equations
# and return the solution in functions
# u,p, and c for velocity, pressure
# and compaction, respectively
###########################################
u,p,c,niter = cavity.momentum_solver(W,a,L,b,bcs,pc="amg")
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
uh.interpolate(straining_flow_2D)
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
