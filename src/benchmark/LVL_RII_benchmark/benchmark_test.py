####################################
## This file calculates the solution 
## for convection-diffusion equations 
## Compares the numerical solution 
## with analytical solution from CH6, 
## Howard Elman. 
##
## Analytical solution for convection
## -diffusion equations with velocity
## u=(0,1) and source term f=0. 
## BCs for both numerical solution 
## and analytical solution are
## c(x,-1)=x, c(x,1)=0, 
## c(-1,y)=-1, c(1,y)=1

## There is still a bug for Expression 
## of Analytical solution. It is due 
## to python or fenics, since it works 
## well, when I calculate Analytical 
## solution expression in an independent
## script, or substitute this expression 
## by a simple equation. 
####################################

from dolfin import *
from mshr import *
import numpy as np
import scipy, sys, math, os, string
#Add the path to Dymms module to the code
#sys.path.insert(0, '../../modules/')
#from mupopp import *

####################################
# Output files
####################################
output_dir     = "output/"
extension      = "pvd"   # "xdmf" or "pvd"
concentration_out = File(output_dir + "concentration." + extension, "compressed")

########################################
# Create initial and boundary conditions
########################################
# Create mesh and define function space
xmin=-1.0
xmax=1.0
ymin=-1.0
ymax=1.0

domain=Rectangle(Point(xmin,ymin),Point(xmax,ymax))
mesh=generate_mesh(domain,60)

# Define function spaces
V = FunctionSpace(mesh, "CG", 1)
W = VectorFunctionSpace(mesh, "CG", 2)

# Output fuction space
X = FunctionSpace(mesh, "CG", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def bottom(x):
    return x[1] < ymin + DOLFIN_EPS
def top(x):
    return x[1] > ymax - DOLFIN_EPS
def left(x):
    return x[0] < xmin + DOLFIN_EPS
def right(x):
    return x[0] > xmax - DOLFIN_EPS

# Define boundary condition
# BCs are from CH6 THE CONVECTION_DIFFUSION EQUATION, 6.1.1
# with u = (0,1) and f = 0
G = Expression(("x[0]"),degree=1)
bc1 = DirichletBC(V, Constant(0.0), top)
bc2 = DirichletBC(V, G, bottom)
bc3 = DirichletBC(V, Constant(-1.0), left)
bc4 = DirichletBC(V, Constant(1.0), right)

bc_c=[bc1,bc2,bc3,bc4]


########################################
# Define and solve the variational problem
########################################
# Define variational problem
v = Expression(("0.0","1.0"),degree=2)
#v0 = Function(W)
#v0.interpolate(Expression(("0.0","1.0"),degree=2))
#v = v0
f = Expression(("0.0"),degree=1)
Pe = Constant(1.0e4)

c = TrialFunction(V)
q = TestFunction(V)

a = dot(grad(c), grad(q))/Pe*dx + dot(v, grad(c))*q*dx
L = f*q*dx

# Compute solution
c_sol = Function(V)
solve(a == L, c_sol, bc_c)

# Output concentration field
c_sol.rename("concentration", "")
concentration_out  << c_sol

############################################
### Compare results with analytical solution
############################################
# Define the analytical solution of c for u = (0,1) and f = 0
class analytical_c(Expression):
    """Calculates analytical solution of concentration"""
    def __init__(self,Pe,element):
        self.Pe=Pe
        self.element=element
    def eval(self,value,x):
	term1 = x[0]*(1.0-np.e**((x[1]-1.0)*Pe))/(1.0-np.e**(-2.0*Pe))

        value[0]=term1

ch0 = Function(V)
analytical_c=analytical_c(Pe =1.0e4,element=V.ufl_element())
ch0.interpolate(analytical_c)
ch = ch0

############################
## output iteration details
############################
# Print error norm of concentration
h = mesh.hmax()
l2_c = errornorm(c_sol,ch,'L2')

def output_info(h,l2_c):
    """This function saves the output of iterations"""
    print '####################################'
    info("\n")
    info("Maximum Cell Diameter:  %g\n" %h)
    info("Concentration solution: L2 norm:%e\n" % l2_c)
    print '####################################'
def output_write(h,l2_c,fname="output/iter.out"):
    """This function saves the output of iterations"""
    file=open(fname,"a")
    file.write("####################################")
    file.write("\n")
    file.write("Maximum Cell Diameter:  %g\n" %h)
    file.write("Concentration solution: L2 norm:%e\n" % l2_c)
    file.write("####################################")
    file.close

output_info(h,l2_c)
output_write(h,l2_c)
