###################################################
## This is a part of Mupopp
## Copyright Saswata Hier-Majumder, July, 2016
## Modified on August 2018
## This program solves an advection diffusion problem
## with Darcy flow, using Dirichlet boundary conditions
## for velocity and concentration
## Modified by Joe Sun, February 2019
# python hdf5_convert.py 
# dolfin-convert test.msh test.xml
####################################################

from fenics import *
from mshr import*
import numpy, scipy, sys, math
#import matplotlib.pyplot as plt
#Add the path to Mupopp module to the code
sys.path.insert(0, '../../modules/')
from mupopp import *


#####################################################
parameters["std_out_all_processes"]=False
#set_log_level(30) #Critical info only
####################################

# Parameters for initializing the object
Da0  = 1.0e-3
Pe0  = 1.0e2
alpha0= 0.01   # beta = alpha0/phi
Fe=0.01
c01_temp=Expression("0.01",degree=1) #Fe
cfl0 = 0.1
phi0 = 0.01
beta=alpha0/phi0
# Parameters for iteration
T0 = 1000.0
dt0 = 1.0e0
out_freq0 = 1

# Parameters for mesh
mesh_density = 60

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
kappa_out = File(output_dir + file_name + "_kappa." + extension, "compressed")

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
    file.write("####################################")
    file.close

output_write(mesh_density,Da0,phi0,Pe0,alpha0,cfl0)

#Define function for source term in Governing equations
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
        g1=x[1]*0.0
        for ii in range(0,20):
            g1+=0.1*np.abs(np.sin(ii*x[1]*np.pi))           
        g = (1.0-tanh(x[0]/0.01))*g1
	if x[1]<=1.55 and x[1]>=1.45: #if x[1]<=2.0 and x[1]>=1.0:
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
        g1=x[1]*0.0
        for ii in range(0,20):
            g1+=0.1*np.abs(np.sin(ii*x[1]*np.pi))
        g = -0.1*g1 # -100*0.1
	if x[1]<=1.55 and x[1]>=1.45: #if x[1]<=2.0 and x[1]>=1.0:
            values[0] = g*n[0]
            values[1] = g*n[1]
    def value_shape(self):
        return (2,)


############################
## Numerical solution
############################

# Define the mesh
xmin = 0.0
xmax = 5.0
ymin = 0.0
ymax = 3.0
#domain = Rectangle(Point(xmin,ymin),Point(xmax,ymax))
#mesh   = generate_mesh(domain,mesh_density)
#mesh = RectangleMesh(Point(xmin, ymin), Point(xmax, ymax), 100, 50)

# Set file names for importing mesh files in .h5 format
mesh_fname = "test.xml"
extension = ".h5"

# Read in the .xml mesh converted to .h5 format
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), "hdf5_" + mesh_fname + extension, "r")
hdf.read(mesh, "/mesh", False)

# Read in the boundaries in .h5 format, called as 'boundaries' to allow for specifying 1, 2, 3 as
# inflow, outflow, walls in the below boundary conditions
#boundaries = FacetFunction("size_t", mesh) # Works but has been depricated in 2017.2.0 and replaced as below
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
hdf.read(boundaries, "/boundaries")
subdomains = MeshFunction("size_t", mesh)
hdf.read(subdomains, "/subdomains")

####################################################
class K(Expression):
    def set_k_values(self,subdomains, k_0, k_1, k_2, k_3):#, *args, **kwargs):
	self.subdomains = subdomains
	self.k_0 = k_0
	self.k_1 = k_1
	self.k_2 = k_2
	self.k_3 = k_3
    def eval_cell(self, values, x, cell):
        if self.subdomains[cell.index] == 42 or self.subdomains[cell.index] == 46:
            values[0] = self.k_0
	elif self.subdomains[cell.index] == 43 or self.subdomains[cell.index] == 47:
            values[0] = self.k_1
	elif self.subdomains[cell.index] == 44 or self.subdomains[cell.index] == 48:
            values[0] = self.k_2		
        else:
            values[0] = self.k_3

# Initialize kappa
kappa = K(degree=0)
kappa.set_k_values(subdomains, 1.0e-15, 1.0e-4, 1.0e-15, 1.0e-4)

####################################################
   

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

def bottom(x):
    return x[1] < DOLFIN_EPS and (x[0]<=3.732 or x[0]>=4.232)
def top(x):
    return x[1] > ymax - DOLFIN_EPS and (x[0]<=2.0 or x[0]>=2.5)
bc4 = DirichletBC(W.sub(0), Constant((0.0,0.0)), top)
bc5 = DirichletBC(W.sub(0), Constant((0.0,0.0)), bottom)

# Define boundary conditions
G=BoundarySource(mesh,element=V)
bc1 = DirichletBC(W.sub(0), G, boundaries, 39) # 39 is the left boundary
bc2 = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, 40)
bc3 = DirichletBC(W.sub(0), Constant((0.0,0.0)), boundaries, 41)

#bc6 = DirichletBC(W.sub(0), Constant((0.1,0.0)), 49)
bc  = [bc1,bc2,bc3]#,bc4]

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


temp3 = Function(X)
temp3.interpolate(kappa)

kappa_out << temp3

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
    a,L = darcy.darcy_advection_rho_posi_random(W,mesh,sol_0,dt,f1=S,K=kappa)
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
