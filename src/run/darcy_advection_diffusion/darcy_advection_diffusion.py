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
#Add the path to Mupopp module to the code
sys.path.insert(0, '../../modules/')
from mupopp import *

#####################################################
def boundary_value(n):
    if n < 10:
        return float(n)/10.0
    else:
        return 1.0
    
# Define function G such that u \cdot n = g
class BoundarySource(Expression):
    def __init__(self, mesh,element):
        self.mesh = mesh
        self.element=element
    def eval_cell(self, values, x, ufl_cell):
        cell = Cell(self.mesh, ufl_cell.index)
        n = cell.normal(ufl_cell.local_facet)
        g = -sin(2.0*x[0]*np.pi)
        values[0] = g*n[0]
        values[1] = g*n[1]
    def value_shape(self):
        return (2,)
xmin=0.0
xmax=4.0
ymin=0.0
ymax=1.0
    
# Define essential boundary
def top_bottom(x):
    return x[1] < DOLFIN_EPS or x[1] > ymax - DOLFIN_EPS
def bottom(x):
    return x[1] < DOLFIN_EPS
def top(x):
    return x[1] > ymax - DOLFIN_EPS

domain=Rectangle(Point(xmin,ymin),Point(xmax,ymax))
mesh=generate_mesh(domain,30)

# Output files for quick visualisation
output_dir     = "output/"
extension      = "pvd"   # "xdmf" or "pvd"

velocity_out   = File(output_dir + "velocity." + extension, "compressed")
pressure_out   = File(output_dir + "pressure." + extension, "compressed")
c_out          = File(output_dir + "concentration." + extension, "compressed")

# Velocity
V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
# Pressure
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# Make a mixed space
W=dolfin.FunctionSpace(mesh, MixedElement([V,Q]))

G=BoundarySource(mesh,element=V)
bc1 = DirichletBC(W.sub(0), G, bottom)
bc2 = DirichletBC(W.sub(0), Constant((0.0,1.0)), top)
bc=[bc1,bc2]

##############Create an object
darcy=DarcyAdvection()

########################
## Solve for Darcy velocity
###########################
a,L=darcy.darcy_bilinear(W,mesh)
sol=Function(W)
solve(a==L,sol,bc)

u,p=sol.split()

velocity_out << u

pressure_out << p

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", 1)
V1= VectorFunctionSpace(mesh,"CG",2)
vel=Function(V1)
vel.interpolate(u)
c0 = Function(Q)
temp = Expression("0.1",degree=1)
c0.interpolate(temp)
# Parameters
T = 20.0
dt = 0.1
t = dt

# Set up boundary condition
g = Constant(boundary_value(0))
bc1 = DirichletBC(Q, Constant(0.0),top)
bc2 = DirichletBC(Q, Constant(1.0),bottom)
bc_c=[bc1,bc2]
sol_c=Function(Q)


i=0
while t - T < DOLFIN_EPS:
    
    a1,L1=darcy.advection_diffusion(Q,c0,vel,dt,mesh)
    solve(a1==L1,sol_c,bc_c)

    c_out << sol_c
    c0 = sol_c

    
    # Move to next interval and adjust boundary condition
    t += dt
    i += 1
    g.assign(boundary_value(int(t/dt)))
