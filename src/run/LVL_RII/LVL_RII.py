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
from mupopp import *

#####################################################
parameters["std_out_all_processes"]=False
#set_log_level(30) #Critical info only
####################################
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
#Define a function h such that c(x,0)=f
class BoundaryConcentration(Expression):
    def __init__(self, mesh,element):
        self.mesh = mesh
        self.element=element
    def eval_cell(self, values, x, ufl_cell):
        f = 0.5+0.5*(1.0-tanh(0.0/0.2))*sin(2.0*x[0]*3.14)
        values[0] = f
        
    def value_shape(self):
        return (1,)

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
mesh=generate_mesh(domain,60)

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
darcy=DarcyAdvection(Da=10.0,phi=0.01,Pe=1.0e6,cfl=0.1)

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
temp.interpolate(Expression("0.1",degree=1))
c01=temp
# Write the initial concentrations into files


# Parameters
T = 50.0
darcy.dt = 0.10
t = darcy.dt

# Set up boundary conditions for component 0
c0_bottom = BoundaryConcentration(mesh,element=Qc)
bc2 = DirichletBC(Q1.sub(0), c0_bottom,bottom)
#bc1 = DirichletBC(Q1.sub(0), Constant(0.0),top)
#bc2 = DirichletBC(Q1.sub(0), Constant(0.5),bottom)
# Set up boundary conditions for component 1
#bc3 = DirichletBC(Q1.sub(1), Constant(0.9),top)
bc4 = DirichletBC(Q1.sub(1), Constant(0.1),bottom)

#bc_c=[bc1,bc2,bc3,bc4]
bc_c=[bc2,bc4]
sol_c=Function(Q1)


i=1
out_freq=10
#Save the dt and time values in an array
dt_array=[]
time_array=[]
while t - T < DOLFIN_EPS:

    #Update the concentration of component 0
    a1,L1=darcy.advection_diffusion_two_component(Q1,c0,vel,mesh)
    solve(a1==L1,sol_c,bc_c)
    c0 = sol_c
    c00,c01=sol_c.split()
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

#Write dt and time into a file
dt_time=np.array([time_array,dt_array])
np.savetxt('dt_time.dat',dt_time)
plt.loglog(time_array,dt_array,'or')
plt.xlabel('time')
plt.ylabel('dt')
#plt.show()
