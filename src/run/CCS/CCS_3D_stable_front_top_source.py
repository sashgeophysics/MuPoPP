###################################################
## This is a part of Mupopp
## Copyright Saswata Hier-Majumder, July, 2016
## Modified on August 2018
## This program solves an advection diffusion problem
## with Darcy flow, using Dirichlet boundary conditions
## for velocity and concentration
## Modified by Joe Sun, February 2019
####################################################

from fenics import *
from mshr import*
import numpy, scipy, sys, math
#import matplotlib.pyplot as plt
#Add the path to Mupopp module to the code
sys.path.insert(0, '../../modules/')
from mupopp import *

import time
start_time = time.time()


#####################################################
parameters["std_out_all_processes"]=False
#set_log_level(30) #Critical info only
####################################

# Parameters for initializing the object
Da0  = 1.0
Pe0  = 200.0
alpha0= 0.01   # beta = alpha0/phi
Fe=0.01
c01_temp=Expression("0.01",degree=1) #Fe
cfl0 = 0.1
phi0 = 0.2
beta=alpha0/phi0
# Parameters for iteration
T0 = 1.0
dt0 = 1.0e-1
out_freq0 = 1

# Parameters for mesh
mesh_density = 50

# Output files for quick visualisation

file_name       =  "Da_%3.2f_Pe_%.1E_beta_%3.2f_Fe_%3.2f"%(Da0,Pe0,beta,Fe)
output_dir     =  "output/"

extension      = "pvd"   # "xdmf" or "pvd"

velocity_out   = File(output_dir + file_name + "_velocity." + extension, "compressed")
pressure_out   = File(output_dir + file_name + "_pressure." + extension, "compressed")
c0_out         = File(output_dir + file_name + "_concentration0." + extension, "compressed")
c1_out         = File(output_dir + file_name + "_concentration1." + extension, "compressed")
initial_c0_out = File(output_dir + file_name + "_initial_c0." + extension, "compressed")
initial_c1_out = File(output_dir + file_name + "_initial_c1." + extension, "compressed")

# Output parameters
def output_write(mesh_density,Da,phi,Pe,alpha,cfl,fname= output_dir + "/a_parameters.out"):
    """This function saves the output of iterations"""
    file=open(fname,"a")
    file.write("####################################")
    file.write("\n")
    file.write("Mesh density:  %g\n" %mesh_density)
    file.write("Da:  %g\n" %Da0)
    file.write("phi:  %g\n" %phi0)
    file.write("Pe:  %g\n" %Pe0)
    file.write("alpha:  %g\n" %alpha0)
    file.write("cfl:  %g\n" %cfl0)
    file.write("dt0:  %g\n" %dt0)
    file.write("T0:  %g\n" %T0)
    file.write("####################################")
    file.close

output_write(mesh_density,Da0,phi0,Pe0,alpha0,cfl0)



############################
## Numerical solution
############################
def mesh_refine(lx,ly,lz,mesh_in):
    m=mesh_in
    x = m.coordinates()
    zmax=np.max(x[:,2])
    #Refine on top and bottom sides
    x[:,2] = (x[:,2] - zmax*0.5) * 2.
    x[:,2] = 0.5 * (np.cos(0.5*np.pi * (x[:,2] - zmax)) + zmax)   
    
    #Scale
    x[:,0] = x[:,0]*lx
    x[:,1] = x[:,1]*ly
    x[:,2] = x[:,2]*lz

    return m
########### 3D######################
# Define the mesh
xmin = 0.0
xmax = 4.0
ymin = 0.0
ymax = 4.0
zmin = 0.0
zmax = 1.0
#domain = Rectangle(Point(xmin,ymin),Point(xmax,ymax))
domain = Box(Point(xmin,ymin,zmin),Point(xmax,ymax,zmax))
mesh1   = generate_mesh(domain,mesh_density)
mesh = mesh_refine(1.0,1.0, 1.0,mesh1)
########### 3D######################
#mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), 100, 50)
#####################

# Mark facets
## Define subdomain for flux calculation
class Plane(SubDomain):
  def inside(self, x, on_boundary):
    return x[2] < DOLFIN_EPS
facets = FacetFunction("size_t", mesh)
Plane().mark(facets, 1)
ds = Measure("ds")[facets]
n = FacetNormal(mesh)
####################
# Define essential boundary
def top_bottom(x):
    return x[2] < DOLFIN_EPS or x[2] > zmax - DOLFIN_EPS
def bottom(x):
    return x[2] < DOLFIN_EPS
def top(x):
    return x[2] > zmax - DOLFIN_EPS
def left_right(x):
    return x[0] < DOLFIN_EPS or x[0] > xmax - DOLFIN_EPS
def left(x):
    return x[0] < DOLFIN_EPS
def right(x):
    return x[0] > xmax - DOLFIN_EPS
# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):
    # Bottom boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(near(x[0], xmin) and near(x[1], ymin) and on_boundary)   
    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        if near(x[0], xmax):
            y[0] = x[0] - xmax
            y[1] = x[1] - ymax
            y[2] = x[2]
        else:
            y[0] = -1000
            y[1] = -1000
            y[2] = -1000
        # Map front boundary (H) to back boundary (G)
        if near(x[1], ymax):
            y[0] = x[0] - xmax
            y[1] = x[1] - ymax
            y[2] = x[2]
        else:
            y[0] = -1000
            y[1] = -1000
            y[2] = -1000
# Create periodic boundary condition
pbc = PeriodicBoundary()        
#Define function for source term in Governing equations

########### 3D######################
class SourceTerm(Expression):
    """ Creates an expression for the source term
    in the advection reaction equations.
    The source term consists of a series of
    sine waves.
    """
    def __init__(self, mesh,element):
        self.mesh = mesh
        self.element=element
    def eval(self, values, x):
        g1=x[0]*0.0
        for ii in range(0,20):
            g1+=0.1*np.abs(np.sin(8.0*ii*x[1]*np.pi))*np.abs(np.sin(8.0*ii*x[0]*np.pi))
            
        g = (1.0-tanh((1.0-x[2])/0.01))*g1
        values[0] = g
    def value_shape(self):
        return (1,)

    
# Define function for BC
class BoundarySource(Expression):
    def __init__(self, mesh,element):
        self.mesh = mesh
        self.element=element
    def eval_cell(self, values, x, ufl_cell):
        cell = Cell(self.mesh, ufl_cell.index)
        n = cell.normal(ufl_cell.local_facet)
        g1=x[0]*0.0
        for ii in range(0,20):
            g1+=0.1*np.abs(np.sin(8.0*ii*x[1]*np.pi)*np.sin(8.0*ii*x[0]*np.pi))
        g = -g1
        values[0] = g*n[0]
        values[1] = g*n[1]
        values[2] = g*n[2]
    def value_shape(self):
        return (3,)
########### 3D######################
############################
## Darcy velocity
############################
# Create FunctionSpaces
# Velocity
V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
# Pressure
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
#Concentration
Qc = FiniteElement("Lagrange",mesh.ufl_cell(), 1)
# Make a mixed space
W = dolfin.FunctionSpace(mesh, MixedElement([V,Q,Qc,Qc]))#, constrained_domain=pbc)
X  = FunctionSpace(mesh,"CG",1)#, constrained_domain=pbc)


# Define boundary conditions
G=BoundarySource(mesh,element=V)
bc1 = DirichletBC(W.sub(0),G, top)
bc2 = DirichletBC(W.sub(2),Constant(0.1), top)
bc  = [bc1]#,bc2]

###########################
## Create an object
###########################
darcy = DarcyAdvection(Da=Da0,phi=phi0,Pe=Pe0,alpha=alpha0,cfl=cfl0)

# Define initial conditions
sol_0 = Function(W)
temp2 = Function(X)
temp2.interpolate(c01_temp)
c01 = temp2
assign(sol_0.sub(3), c01)

#c0_initial,c1_initial=c0.split()
#initial_c0_out << c0_initial
#initial_c1_out << c1_initial

###########################
## Solve for Darcy velocity
###########################
sol = Function(W)
# Parameters for iteration
T = T0
dt = dt0
t = dt
flux=np.array([])
time_array=np.array([])
i = 1
out_freq = out_freq0
S=SourceTerm(mesh,element=Qc)

while t - T < DOLFIN_EPS:
    # Update the concentration of component 0
    a,L = darcy.advection_diffusion_two_component(W,mesh,sol_0,dt,f1=S,zh=Constant((0.0,0.0,1.0)),gam_rho=-1.0,rho_0=0.0)
    solve(a==L,sol,bc)
    sol_0 = sol
    u0,p0,c00,c01 = sol.split()

    if i % out_freq == 0:
	u0.rename("velocity","")
	velocity_out << u0
	p0.rename("pressure","")
	pressure_out << p0
        c00.rename("[CO3]","")
        c0_out << c00
        c01.rename("[Fe]","")
        c1_out << c01
        ## Calculate flux
        flux1 = assemble(c00*phi0*dot(u0, n)*ds(1))
        flux= np.append(flux,flux1)
        time_array=np.append(time_array,t)
        #print "flux 1: ", flux_1
    # Move to next interval and adjust boundary condition
    info("time t =%g\n" %t)
    info("iteration =%g\n" %i)
    #print 'iteration',i
    t += dt
    i += 1
flux_file=output_dir + file_name + "_flux.csv"
np.savetxt(flux_file,(time_array,flux),delimiter=',')
info("Finished execution in %s seconds ---" % (time.time() - start_time))
