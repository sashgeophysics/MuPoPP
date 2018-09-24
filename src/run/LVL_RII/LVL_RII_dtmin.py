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
from dolfin import *
import numpy, scipy, sys, math
#import core
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
        g = -np.abs(sin(2.0*x[0]*np.pi))
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
initial_c0_out   = File(output_dir + "initial_c0." + extension, "compressed")
# Velocity
V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
# Pressure
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# Make a mixed space
W=dolfin.FunctionSpace(mesh, MixedElement([V,Q]))#, constrained_domain=pbc)

G=BoundarySource(mesh,element=V)
bc1 = DirichletBC(W.sub(0), G, bottom)
bc2 = DirichletBC(W.sub(0), Constant((0.0,0.001)), top)
bc=[bc1,bc2]

##############Create an object
darcy=DarcyAdvection(Da=1.0,phi=0.01,Pe=1.0e6)

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
vtemp=Function(V1)
vtemp.interpolate(Expression(("0.0","0.001"),degree=2))
vel = vtemp #.interpolate(vtemp)

# Create functions for initial conditions
c0 = Function(Q1)
c00,c01=c0.split()
temp = Function(X)
temp.interpolate(Expression("0.5+0.5*(1.0-tanh(x[1]/0.2))*sin(2.0*x[0]*3.14)",degree=2))
c00=temp
c00.rename("[CO3]","")
initial_c0_out << c00
temp.interpolate(Expression("0.9",degree=1))
c01=temp
# Write the initial concentrations into files


# Parameters
T = 100.0
#dt = 0.010

#============================================================================

def u_max(U, cylinder_mesh):
    """Return |u|_max for a U = (u, p) systems"""

    mesh   = U.function_space().mesh()
    V      = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    v      = TestFunction(V)

    if cylinder_mesh:
        u, p, o = U.split()
    else:
        u, p    = U.split()

#    volume = 1.0e-4   
#    volume = v.cell().volume
#    volume = v.function_space().mesh().Cell().volume()
    testfunctionmesh = v.function_space().mesh()
    celldiameter = CellDiameter(testfunctionmesh)
    volume = celldiameter**3
    L      = v*sqrt(dot(u, u))/volume*dx
    b      = assemble(L)

    return b.norm("linf")

def compute_dt(U, cfl, h_min, cylinder_mesh):
    """Compute time step dt, 
	cfl and cylinder_mesh are from param.cfg, 
	h_min is from"""

    umax   = u_max(U, cylinder_mesh)

    return cfl*h_min/max(1.0e-6, umax)

cfl     = 0.1                # cfl number from param.cfg
cylinder_mesh = 0            # flag for having a cylinder in the mesh with torque-free BC
#h_min = MPI.min(mesh.hmin()) 
h_min = mesh.hmin()          # Smallest element size. Used to determine time step
U=sol

dt = compute_dt(U, cfl, h_min, cylinder_mesh)

print dt 
#==============================================================================

t = dt

# Set up boundary conditions for component 0
bc1 = DirichletBC(Q1.sub(0), Constant(0.0),top)
bc2 = DirichletBC(Q1.sub(0), Constant(0.5),bottom)
# Set up boundary conditions for component 1
bc3 = DirichletBC(Q1.sub(1), Constant(0.9),top)
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
    
