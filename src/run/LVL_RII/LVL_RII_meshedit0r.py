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
init_mesh=generate_mesh(domain,30)


#########################################################
def wedge_mesh_RII(n=4):
    """This function uses a wedge shaped mesh from a box
    with lengths dimx and dimy, and degree of refinement n"""
    ######################################################
    ## First we use a mesh editor to create a slanted mesh
    editor = MeshEditor()
    mesh0 = Mesh()

    gdim = init_mesh.geometry().dim()
    tdim = init_mesh.topology().dim()
    c_type = init_mesh.type()
    c_str = c_type.type2string(c_type.cell_type())
    print ('gdim=',gdim)
    print ('tdim=',tdim)
    print ('c_str=',c_str)

    editor.open(mesh0, c_str, tdim, gdim)  # top. and geom. dimension are both 2

    editor.init_vertices(28)  # number of vertices
    editor.init_cells(32)     # number of cells

    editor.add_vertex(0, np.array([0.0, 1.00]))
    editor.add_vertex(1, np.array([0.0, 0.75]))
    editor.add_vertex(2, np.array([0.5, 1.00]))
    editor.add_vertex(3, np.array([0.5, 0.75]))
    editor.add_vertex(4, np.array([1.0, 1.00]))
    editor.add_vertex(5, np.array([1.0, 0.75]))
    editor.add_vertex(6, np.array([1.5, 1.00]))
    editor.add_vertex(7, np.array([1.5, 0.75]))
    editor.add_vertex(8, np.array([2.0, 1.00]))
    editor.add_vertex(9, np.array([2.0, 0.75]))
    editor.add_vertex(10, np.array([2.5, 1.00]))
    editor.add_vertex(11, np.array([2.5, 0.75]))
    editor.add_vertex(12, np.array([3.0, 1.00]))
    editor.add_vertex(13, np.array([3.0, 0.75]))
    editor.add_vertex(14, np.array([3.5, 1.00]))
    editor.add_vertex(15, np.array([3.5, 0.75]))
    editor.add_vertex(16, np.array([4.0, 1.00]))
    editor.add_vertex(17, np.array([4.0, 0.75]))

    editor.add_vertex(18, np.array([0.0, 0.5]))
    editor.add_vertex(19, np.array([1.0, 0.5]))
    editor.add_vertex(20, np.array([2.0, 0.5]))
    editor.add_vertex(21, np.array([3.0, 0.5]))
    editor.add_vertex(22, np.array([4.0, 0.5]))

    editor.add_vertex(23, np.array([0.0, 0.0]))
    editor.add_vertex(24, np.array([1.0, 0.0]))
    editor.add_vertex(25, np.array([2.0, 0.0]))
    editor.add_vertex(26, np.array([3.0, 0.0]))
    editor.add_vertex(27, np.array([4.0, 0.0]))


    editor.add_cell(0, np.array([0, 1, 3], dtype=np.uintp))
    editor.add_cell(1, np.array([0, 2, 3], dtype=np.uintp))
    editor.add_cell(2, np.array([2, 3, 5], dtype=np.uintp))
    editor.add_cell(3, np.array([2, 4, 5], dtype=np.uintp))
    editor.add_cell(4, np.array([4, 5, 7], dtype=np.uintp))
    editor.add_cell(5, np.array([4, 6, 7], dtype=np.uintp))
    editor.add_cell(6, np.array([6, 7, 9], dtype=np.uintp))
    editor.add_cell(7, np.array([6, 8, 9], dtype=np.uintp))
    editor.add_cell(8, np.array([8, 9, 11], dtype=np.uintp))
    editor.add_cell(9, np.array([8, 10, 11], dtype=np.uintp))
    editor.add_cell(10, np.array([10, 11, 13], dtype=np.uintp))
    editor.add_cell(11, np.array([10, 12, 13], dtype=np.uintp))
    editor.add_cell(12, np.array([12, 13, 15], dtype=np.uintp))
    editor.add_cell(13, np.array([12, 14, 15], dtype=np.uintp))
    editor.add_cell(14, np.array([14, 15, 17], dtype=np.uintp))
    editor.add_cell(15, np.array([14, 16, 17], dtype=np.uintp))

    editor.add_cell(16, np.array([1, 18, 19], dtype=np.uintp))
    editor.add_cell(17, np.array([1, 5, 19], dtype=np.uintp))
    editor.add_cell(18, np.array([5, 19, 20], dtype=np.uintp))
    editor.add_cell(19, np.array([5, 9, 20], dtype=np.uintp))
    editor.add_cell(20, np.array([9, 20, 21], dtype=np.uintp))
    editor.add_cell(21, np.array([9, 13, 21], dtype=np.uintp))
    editor.add_cell(22, np.array([13, 21, 22], dtype=np.uintp))
    editor.add_cell(23, np.array([13, 17, 22], dtype=np.uintp))

    editor.add_cell(24, np.array([18, 23, 24], dtype=np.uintp))
    editor.add_cell(25, np.array([18, 19, 24], dtype=np.uintp))
    editor.add_cell(26, np.array([19, 24, 25], dtype=np.uintp))
    editor.add_cell(27, np.array([19, 20, 25], dtype=np.uintp))
    editor.add_cell(28, np.array([20, 25, 26], dtype=np.uintp))
    editor.add_cell(29, np.array([20, 21, 26], dtype=np.uintp))
    editor.add_cell(30, np.array([21, 26, 27], dtype=np.uintp))
    editor.add_cell(31, np.array([21, 22, 27], dtype=np.uintp))

    editor.close()
#    print ('Dolfin Version',dolfin.dolfin_version())
    mesh=refine(init_mesh)
    # Refine the mesh n times
    
    for x in range(0, n):
        mesh=refine(mesh)
    return mesh
####################################################

mesh = wedge_mesh_RII()

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
T = 20.0
dt = 5.0
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
    
