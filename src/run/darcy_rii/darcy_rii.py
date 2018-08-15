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

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):
    # Bottom boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool( near(x[0], xmin) and on_boundary)
    
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

domain=Rectangle(Point(xmin,ymin),Point(xmax,ymax))
mesh=generate_mesh(domain,30)

# Output files for quick visualisation
output_dir     = "output/"
extension      = "pvd"   # "xdmf" or "pvd"

velocity_out   = File(output_dir + "velocity." + extension, "compressed")
pressure_out   = File(output_dir + "pressure." + extension, "compressed")
c0_out          = File(output_dir + "concentration0." + extension, "compressed")
c1_out          = File(output_dir + "concentration1." + extension, "compressed")

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
darcy=DarcyAdvection(Da=100.0,phi=0.01,Pe=1.0e4)

########################
## Solve for Darcy velocity
###########################
a,L=darcy.darcy_bilinear(W,mesh)
sol=Function(W)
solve(a==L,sol,bc)

u,p=sol.split()

velocity_out << u

pressure_out << p
############################
## Concentrations
############################
# Create finite elements
Qc = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# Make a mixed space containing two concentrations
Q1=dolfin.FunctionSpace(mesh, MixedElement([Qc,Qc]),constrained_domain=pbc)
X = FunctionSpace(mesh,"CG",1, constrained_domain=pbc)
V1= VectorFunctionSpace(mesh,"CG",2, constrained_domain=pbc)
vel=Function(V1)
vel.interpolate(u)

# Create functions for initial conditions
c0 = Function(Q1)
c00,c01=c0.split()
temp = Function(X)
temp.interpolate(Expression("0.1",degree=1))
c00=temp
temp.interpolate(Expression("0.9",degree=1))
c01=temp
# Write the initial concentrations into files
#c0_out << c00
#c1_out << c01
# Parameters
T = 1.0
dt = 0.001
t = dt

# Set up boundary conditions for component 0
bc1 = DirichletBC(Q1.sub(0), Constant(0.0),top)
bc2 = DirichletBC(Q1.sub(0), Constant(1.0),bottom)
# Set up boundary conditions for component 1
bc3 = DirichletBC(Q1.sub(1), Constant(1.0),top)
bc4 = DirichletBC(Q1.sub(1), Constant(0.0),bottom)

bc_c=[bc1,bc2,bc3,bc4]
sol_c=Function(Q1)


i=1
out_freq=10

while t - T < DOLFIN_EPS:

    #Update the concentration of component 0
    a1,L1=darcy.advection_diffusion_two_component(Q1,c0,vel,dt,mesh)
    solve(a1==L1,sol_c,bc_c)
    c0 = sol_c
    c00,c01=sol_c.split()
    if i % out_freq == 0:
        c00.rename("[CO3]","")
        c0_out << c00
        c01.rename("[Fe]","")
        c1_out << c01
    

    
    # Move to next interval and adjust boundary condition
    t += dt
    i += 1
    #g.assign(boundary_value(int(t/dt)))
