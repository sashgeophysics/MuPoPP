from fenics import *
from mshr import*
import numpy, scipy, sys, math
#Add the path to Mupopp module to the code
sys.path.insert(0, '../../modules/')
from mupopp import *

xmin=0.0
xmax=4.0
ymin=0.0
ymax=1.0
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

initial_c0_out   = File(output_dir + "initial_c0." + extension, "compressed")

Q = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)
c0 = Function(Q)
temp = Expression("0.5*(1.0-tanh(x[1]/0.2))",degree=1)
c0.interpolate(temp)




initial_c0_out << c0
