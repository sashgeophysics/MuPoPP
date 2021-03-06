from fenics import *

p1=Point(0,0,0)
p2=Point(1,1,1)
mesh=BoxMesh(p1,p2,6,6,6)

V_el = VectorElement("Lagrange", mesh.ufl_cell(), 2)
T1   = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Q_el = MixedElement([T1,T1])

V    = FunctionSpace(mesh,V_el)
Q    = FunctionSpace(mesh,Q_el)

u    = Function(V)
sol   = Function(Q)
sol0  = Function(Q)

c,phi=split(sol)
c0,phi0=split(sol0)

#Define constants
Da=1.0
Gam=1.0
Pe=1.0
B=1.0e3
R=0.1


q,w   = TestFunctions(Q)

u = Constant((0.0,0.1,0.0))

#Create the initial values
X     = FunctionSpace(mesh,T1)
temp  = Expression("0.1*x[0]*x[0]-x[1]*x[2]",degree=2)
phi0  = interpolate(temp,X)
temp  = Expression("0.01",degree=1)
c0    = interpolate(temp,X)

dt=0.01

File("c0.pvd") << c0
File("phi0.pvd") << phi0

#Set up the variational problem
phim=0.5*(phi+phi0)
cm=0.5*(c+c0)

f1=Da*Gam*c0/(1.0-R)
f2=(1.0-c0)*Da*Gam*c0/(1.0-R)/phi0
a=w*(phi-phi0+dt*(phim*div(u)+inner(u,grad(phim))))*dx\
   +q*(c-c0 + dt*(cm*div(u)+inner(u,grad(cm))\
                  +inner(grad(cm),grad(q))/Pe))*dx
L=dt*f1*w*dx+dt*q*f2*dx

###############
## solvers stuff
###################

A=assemble(a)
b=assemble(L)
solve(A,sol.vector(),b)
c,phi=sol.split()
