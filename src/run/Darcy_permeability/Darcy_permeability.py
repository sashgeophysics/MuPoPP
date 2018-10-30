###################################################
## This is a part of Mupopp
## Copyright Saswata Hier-Majumder, July, 2016
## Modified on August 2018
## This program solves an advection diffusion problem
## with Darcy flow, using Dirichlet boundary conditions
## for velocity and concentration
####################################################

from fenics import *
from mshr import*
import numpy, scipy, sys, math
import matplotlib.pyplot as plt
#Add the path to Mupopp module to the code
sys.path.insert(0, '../../modules/')
#from mupopp import *

# muppop_Joe contains the newest equations
from mupopp_Joe import *

#####################################################
parameters["std_out_all_processes"]=False
#set_log_level(30) #Critical info only
####################################

# Parameters for initializing the object
Da0  = 10.0
phi0 = 0.01
Pe0  = 1.0e6
cfl0 = 0.1

# Parameters for mesh
mesh_density = 30

# Output files for quick visualisation
output_dir     = "output/"
extension      = "pvd"   # "xdmf" or "pvd"

velocity_out   = File(output_dir + "velocity." + extension, "compressed")
pressure_out   = File(output_dir + "pressure." + extension, "compressed")
c0_out         = File(output_dir + "concentration0." + extension, "compressed")
c1_out         = File(output_dir + "concentration1." + extension, "compressed")
initial_c0_out = File(output_dir + "initial_c0." + extension, "compressed")
    
# Define function G such that u \cdot n = g
class BoundarySource(Expression):
    def __init__(self, mesh,element):
        self.mesh = mesh
        self.element=element
    def eval_cell(self, values, x, ufl_cell):
        cell = Cell(self.mesh, ufl_cell.index)
        n = cell.normal(ufl_cell.local_facet)
        g = -np.abs(sin(2.0*x[0]*np.pi))
        values[0] = g*n[0]
        values[1] = g*n[1]
	values[2] = g*n[2]
    def value_shape(self):
        return (3,)

############################
## Numerical solution
############################

# Define the mesh
xmin = 0.0
xmax = 4.0
ymin = 0.0
ymax = 1.0
zmin = 0.0
zmax = 1.0
geometry = Box(Point(xmin,ymin,zmin),Point(xmax,ymax,zmax))
mesh   = generate_mesh(geometry, mesh_density, "cgal")
#mesh = generator.generate(CSGCGALDomain3D(domain))
   
# Define essential boundary
def back(x):
    return x[0] < xmin + DOLFIN_EPS
def front(x):
    return x[0] > xmax - DOLFIN_EPS
def left(x):
    return x[1] < ymin + DOLFIN_EPS
def right(x):
    return x[1] > ymax - DOLFIN_EPS
def bottom(x):
    return x[2] < zmin + DOLFIN_EPS
def top(x):
    return x[2] > zmax - DOLFIN_EPS

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):
    # Back boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(near(x[0], xmin) and on_boundary)   
    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        if near(x[0], xmax):
            y[0] = x[0] - xmax
            y[1] = x[1]            
        else:
            y[0] = -1000
            y[1] = -1000
# Create periodic boundary condition
pbc = PeriodicBoundary()            

# Create FunctionSpaces
# Velocity
V = VectorElement("Lagrange", mesh.ufl_cell(), 3)
# Pressure
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# Make a mixed space
W = dolfin.FunctionSpace(mesh, MixedElement([V,Q]))#, constrained_domain=pbc)

# Define boundary conditions
G=BoundarySource(mesh,element=V)
bc1 = DirichletBC(W.sub(0), G, back)
bc2 = DirichletBC(W.sub(0), Constant((0.0,0.0,0.001)), front)
bc  = [bc1,bc2]

###########################
## Create an object
###########################
darcy = DarcyAdvection(Da=Da0,phi=phi0,Pe=Pe0,cfl=cfl0)

###########################
## Solve for Darcy velocity
###########################
a,L = darcy.darcy_bilinear(W,mesh,zh=Constant((0.0,0.0,1.0)))
sol = Function(W)
solve(a==L,sol,bc)

u,p = sol.split()

velocity_out << u
pressure_out << p
