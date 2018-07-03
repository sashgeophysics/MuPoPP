import time
import os
import math
from dolfin import *

# get file name
fileName = os.path.splitext(__file__)[0]

parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True

# Parameters
Pe = Constant(1e10)
t_end = 10 
dt = 0.1

# Create mesh and define function space
mesh = RectangleMesh(0, 0, 1, 1, 40, 40, 'crossed')

# Define function spaces
V = FunctionSpace(mesh, "CG", 1)

#ic= Expression("((pow(x[0]-0.25,2)+pow(x[1]-0.25,2))<0.2*0.2)?(-25*((pow(x[0]-0.25,2)+pow(x[1]-0.25,2))-0.2*0.2)):(0.0)")
ic= Expression("((pow(x[0]-0.3,2)+pow(x[1]-0.3,2))<0.2*0.2)?(1.0):(0.0)", domain=mesh)

b = Expression(("-(x[1]-0.5)","(x[0]-0.5)"), domain=mesh)

bc=DirichletBC(V,Constant(0.0),DomainBoundary())
	
# Define unknown and test function(s)
v = TestFunction(V)
u = TrialFunction(V)

u0 = Function(V)
u0 = interpolate(ic,V )

# STABILIZATION
h = CellSize(mesh)
n = FacetNormal(mesh)
theta = Constant(1.0)

# Define variational forms
a0=(1.0/Pe)*inner(grad(u0), grad(v))*dx + inner(b,grad(u0))* v *dx
a1=(1.0/Pe)*inner(grad(u), grad(v))*dx + inner(b,grad(u))* v *dx

A = (1/dt)*inner(u, v)*dx - (1/dt)*inner(u0,v)*dx + theta*a1 + (1-theta)*a0

F = A

# Create files for storing results
file = File("results_%s/u.xdmf" % (fileName))

u = Function(V)
ffc_options = {"optimize": True, "quadrature_degree": 8}
problem = LinearVariationalProblem(lhs(F),rhs(F), u, [bc], form_compiler_parameters=ffc_options)
solver  = LinearVariationalSolver(problem)


u.assign(u0)
u.rename("u", "u")

# Time-stepping
t = 0.0

file << u

while t < t_end:

	print "t =", t, "end t=", t_end

	# Compute
	solver.solve()
        plot(u)
	# Save to file
	file << u

	# Move to next time step
	u0.assign(u)
	t += dt
