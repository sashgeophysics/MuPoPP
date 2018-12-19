###################################################
## This is a part of Mupopp
## Copyright Saswata Hier-Majumder, July, 2016
## Modified on December 2018
## This program solves an advection diffusion problem
## with Darcy flow, using Dirichlet boundary conditions
## for velocity and concentration
####################################################

from fenics import *
from mshr import*
import numpy, scipy, sys, math
#import matplotlib.pyplot as plt
#Add the path to Mupopp module to the code
sys.path.insert(0, '../../modules/')
#from mupopp import *

# muppop_Joe contains the newest equations
from mupopp import *

class BoundaryConcentration(Expression):
    def __init__(self, mesh, element):
        self.mesh = mesh
        self.element=element
    def eval_cell(self, values, x, ufl_cell):
        f = 0.05*(1.0-tanh(0.0/0.2))*(1.0+sin(2.0*x[0]*3.14))/2.0
        values[0] = f       
    def value_shape(self):
        return (1,)
#####################################################
parameters["std_out_all_processes"]=False
#set_log_level(30) #Critical info only
# Parameters for initializing the object
Da0  = 1.0
phi0 = 0.01
Pe0  = 1.0e4
alpha0= 0.01   # beta = alpha0/phi
cfl0 = 0.1
v_temp=Expression(("0.0","0.1"),degree=2)
c01_temp=Expression("0.01",degree=1)

# Parameters for iteration
T0 = 0.1
dt0 = 1.0e-4
out_freq0 = 5
####################################
# Output files for quick visualisation
output_dir     = "output/"
extension      = "pvd"   # "xdmf" or "pvd"

velocity_out   = File(output_dir + "velocity." + extension, "compressed")
pressure_out   = File(output_dir + "pressure." + extension, "compressed")
c0_out          = File(output_dir + "concentration0." + extension, "compressed")
c1_out          = File(output_dir + "concentration1." + extension, "compressed")
initial_c0_out   = File(output_dir + "initial_c0." + extension, "compressed")
initial_c1_out   = File(output_dir + "initial_c1." + extension, "compressed")
############################
## Numerical solution
############################

# Define the mesh
xmin = 0.0
xmax = 4.0
ymin = 0.0
ymax = 1.0
domain = Rectangle(Point(xmin,ymin),Point(xmax,ymax))
# Parameters for mesh
mesh_density = 60
mesh   = generate_mesh(domain,mesh_density)
#mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), 100, 50)
    
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
####################################################
# Create FunctionSpaces
Qc = FiniteElement("Lagrange",mesh.ufl_cell(), 1)
QM = dolfin.FunctionSpace(mesh, MixedElement([Qc,Qc]))#,constrained_domain=pbc)
X  = FunctionSpace(mesh,"CG",1)#, constrained_domain=pbc)
Vc = VectorFunctionSpace(mesh,"CG",2)#, constrained_domain=pbc)

# Define the constant velocity
vel = Function(Vc)
vtemp = Function(Vc)
vtemp.interpolate(v_temp)
vel = vtemp

###########################
## Create an object
###########################
darcy = DarcyAdvection(Da=Da0,phi=phi0,Pe=Pe0,alpha=alpha0,cfl=cfl0)

############################
## Define initial conditions 
############################
# Create functions for initial conditions
c0 = Function(QM)

# Functions for components
temp1 = Function(X)
temp1.interpolate(Expression("0.5+0.5*(1.0-tanh(x[1]/0.2))*sin(2.0*x[0]*3.14)",degree=2))
c00 = temp1
temp2 = Function(X)
temp2.interpolate(c01_temp)
c01 = temp2

# Assign the components
#assign(c0.sub(0), c00)
assign(c0.sub(1), c01)

#assign(c0, [c01,c01])

#c_initial = InitialConcentration(element=MixedElement([Qc,Qc]))
#c0.interpolate(c_initial)

c0_initial,c1_initial=c0.split()
initial_c0_out << c0_initial
initial_c1_out << c1_initial

############################
## Define boundary conditions
############################
# Set up boundary conditions for component 0
c0_bottom = BoundaryConcentration(mesh,element=Qc)
bc1 = DirichletBC(QM.sub(0), Constant(0.0),top)
bc2 = DirichletBC(QM.sub(0), c0_bottom,bottom)

# Set up boundary conditions for component 1
bc3 = DirichletBC(QM.sub(1), Constant(0.1),top)
bc4 = DirichletBC(QM.sub(1), Constant(0.1),bottom)

#bc_c = [bc1,bc2,bc3,bc4]
bc_c = [bc2]

###########################
## Solve for Concentrations
###########################
sol_c = Function(QM)
# Parameters for iteration
T = T0
darcy.dt = dt0
t = darcy.dt

i = 1
out_freq = out_freq0
# Save the dt and time values in an array
dt_array = []
time_array = []
while t - T < DOLFIN_EPS:

    # Update the concentration of component 0
    ac,Lc = darcy.advection_diffusion_two_component(QM,c0,vel,mesh)
    solve(ac==Lc,sol_c)#,bc_c) # tranfer the DirichletBC into a source term
    c0 = sol_c
    #c0.assign(sol_c) # 'assign' is verified to be equal to '=' here
    c00,c01 = sol_c.split()
    if i % out_freq == 0:
        c00.rename("[CO3]","")
        c0_out << c00
        c01.rename("[Fe]","")
        c1_out << c01
    # Move to next interval and adjust boundary condition
    dt_array.append(darcy.dt)
    time_array.append(t)
    info("time t =%g\n" %t)
    info("iteration =%g\n" %i)
    #print 'iteration',i
    t += darcy.dt
    i += 1
