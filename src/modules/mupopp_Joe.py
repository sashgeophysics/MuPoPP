""" This is a simplified version of MuPPoP just for LVL_RII case.

The solvers and boundary conditions should be set within the script 
files calling on the module functions. 

Works with Dolfin 2017.2.0
Python 2.7
"""
from fenics import *
from dolfin import *
import numpy as np
import scipy, sys, math,os,string
import datetime

###################################################################
### Advection diffusion equation in Darcy flow
###################################################################
class DarcyAdvection():
    """
    This class solves for a simple advection-diffusion
    equation for a single or multicomponent flow, the governing
    PDEs for Darcy flow are
    
    """
    
    def __init__(self,Pe=100,Da=10.0,phi=0.01,cfl=1.0e-2,dt=1.0e-2):
        """Initiates the class. Inherits nondimensional numbers
        from compaction, only need to add Pe"""
        self.Pe=Pe
        self.Da=Da
        self.phi=phi
        self.cfl=cfl
        self.dt=dt
    def darcy_bilinear(self,W,mesh,K=0.1,zh=Constant((0.0,1.0))):
        """            
        """
        U = TrialFunction(W)
        (v, q) = TestFunctions(W)
        u, p   = split(U)
        
        zhat=zh
        
        f=Expression("1.0",degree=1)
        # Define the variational form

        a = (inner(self.phi*u,v)-K*div(v)*p+div(u)*q)*dx # self.phi*u
        L = K*f*inner(v,zhat)*dx
        
        return(a,L)
    
    def advection_diffusion(self,Q, u0, velocity, dt,mesh):
        
        f  = Expression("0.0",degree=1)       
        h = CellDiameter(mesh)
        # Parameters

        # Test and trial functions
        u, v = TrialFunction(Q), TestFunction(Q)
        
        # Mid-point solution
        u_mid = 0.5*(u0 + u)
    
        f = self.Da*u0 # need to modify for first order reaction
        # Residual
        r = u - u0 + dt*(dot(velocity, grad(u_mid)) - div(grad(u_mid))/self.Pe+f)
        
        # Galerkin variational problem
        F = v*(u - u0)*dx + dt*(v*dot(velocity, grad(u_mid))*dx\
                + dot(grad(v), grad(u_mid)/self.Pe)*dx)+dt*self.Da*u0*v*dx
        
        # Add SUPG stabilisation terms
        vnorm = sqrt(dot(velocity, velocity))
        F += (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx 
        
        return lhs(F), rhs(F)
    def advection_diffusion_two_component_nonadaptive(self,Q, c_prev, velocity, dt,mesh):
        """ This function builds the bilinear form for component advection
        diffusion for componet one. The source term depends on the concentration
        of both components. Concentration of the other component should be
        calculated outside this function after solving the bilinear form from
        this step"""
             
        h = CellDiameter(mesh)

        # TrialFunctions and TestFunctions
        U = TrialFunction(Q)
        (v, q) = TestFunctions(Q)
        # u and c are the trial functions for the next time step
        # u for comp 0 and c comp1 
        u, c   = split(U)

        # u0 (component 0) and c0(component 1)
        # are known values from the previous time step
        u0 ,c0 = split(c_prev)

        # Mid-point solution for comp 0
        u_mid = 0.5*(u0 + u)

        # First order reaction term
        f = self.Da*u0*c0

        # Galerkin variational problem
        # Residual
        r = u - u0 + dt*(dot(velocity, grad(u_mid)) - div(grad(u_mid))/self.Pe\
		+ f/self.phi) + c - c0 + dt*f/(1-self.phi) #2.f/self.phi 3.dt*f/(1-self.phi)
	# Add SUPG stabilisation terms
        vnorm = sqrt(dot(velocity, velocity))
        alpha_SUPG = self.Pe*vnorm*h/2.0

        # Brookes and Hughes
        coth = (np.e**(2.0*alpha_SUPG)+1.0)/(np.e**(2.0*alpha_SUPG)-1.0)
        term1_SUPG = 0.5*h*(coth-1.0/alpha_SUPG)/vnorm
        # Sendur 2018
        tau_SUPG1 = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h)
        # Codina 1997 eq. 114
        tau_SUPG2 = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h+self.Da/self.phi) #4.self.Da/self.phi

        term_SUPG = tau_SUPG2*dot(velocity, grad(v))*r*dx

	F = v*(u - u0)*dx + dt*(v*dot(velocity, grad(u_mid))*dx\
                + dot(grad(v), grad(u_mid)/self.Pe)*dx) + dt*f/self.phi*v*dx\
                + q*(c - c0)*dx + dt*f/(1-self.phi)*q*dx  #5.f/self.phi 6.dt*f/(1-self.phi)
        F += term_SUPG
        
        return lhs(F), rhs(F)

    def advection_diffusion_two_component(self,Q, c_prev, velocity,mesh):
        """ This function builds the bilinear form for component advection
        diffusion for component one. The source term depends on the concentration
        of both components. Concentration of the other component should be
        calculated outside this function after solving the bilinear form from
        this step"""
             
        h = CellDiameter(mesh)

        # TrialFunctions and TestFunctions
        U = TrialFunction(Q)
        (v, q) = TestFunctions(Q)

        # u and c are the trial functions for the next time step
        # u for comp 0 and c comp1 
        u, c   = split(U)

        # u0 (component 0) and c0(component 1)
        # are known values from the previous time step
        u0 ,c0 = split(c_prev)

        # Mid-point solution for comp 0
        u_mid = 0.5*(u0 + u)

        # First order reaction term
        f = self.Da*u0*c0

	##################################################
        # Adaptive time-stepping added September 2018
        # Right hand side of advection equation
        # u.grad(c)-div(grad(c))/Pe+f

        # Evaluate this term from the last time step
        advect_term = dot(velocity, grad(u0)) - div(grad(u0))/self.Pe + f/self.phi #1.f/self.phi
        # Create a DG function sapce to evaluate the values of this
        DG = FunctionSpace(mesh, "DG", 0)
        #Save it into a function 
        advect_rhs=project(advect_term,DG)
        # Now evaluate the maximum value of this term
        ad_max=advect_rhs.vector().max()
        #Compute dt for this time step such that
        #dt*ad_max<=CFL
        if np.abs(ad_max)>self.cfl:
            self.dt=self.cfl/ad_max       
        dt=self.dt
        #print 'dt',dt,'ad_max',ad_max
        #####End adaptive time stepping
	##################################################
        
        # Galerkin variational problem
        # Residual
        r = u - u0 + dt*(dot(velocity, grad(u_mid)) - div(grad(u_mid))/self.Pe\
		+ f/self.phi) + c - c0 + dt*f/(1-self.phi) #2.f/self.phi 3.dt*f/(1-self.phi)
        # Add SUPG stabilisation terms
        vnorm = sqrt(dot(velocity, velocity))
        alpha_SUPG = self.Pe*vnorm*h/2.0

        # Brookes and Hughes
        coth = (np.e**(2.0*alpha_SUPG)+1.0)/(np.e**(2.0*alpha_SUPG)-1.0)
        term1_SUPG = 0.5*h*(coth-1.0/alpha_SUPG)/vnorm
        # Sendur 2018
        tau_SUPG1 = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h)
        # Codina 1997 eq. 114
        tau_SUPG2 = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h+self.Da/self.phi) #4.self.Da/self.phi

        term_SUPG = tau_SUPG2*dot(velocity, grad(v))*r*dx

        F = v*(u - u0)*dx + dt*(v*dot(velocity, grad(u_mid))*dx\
                + dot(grad(v), grad(u_mid)/self.Pe)*dx) + dt*f/self.phi*v*dx\
                + q*(c - c0)*dx + dt*f/(1-self.phi)*q*dx  #5.f/self.phi 6.dt*f/(1-self.phi)
        F += term_SUPG
        
        return lhs(F), rhs(F)

########################################################
## A class for LVL_RII_benchmark
#######################################################
class DarcyAdvection_benchmark():
    """
    This class solves for mumerical solution of LVL_RII_benchmark
    
    """  
    def __init__(self,Pe=100,Da=10.0):
        self.Pe=Pe
        self.Da=Da
    def LVL_RII_benchmark(self,Q,mesh):
        """ This function is built fot LVL_RII_benchmark. Equations are 
	from CH6, Howard Elman. The source term is zero."""

	# Define variational problem
	v = Expression(("0.0","1.0"),degree=2)
	h = CellDiameter(mesh)
	#v0 = Function(Q)
	#v0.interpolate(Expression(("0.0","1.0"),degree=2))
	#v = v0
	f = Expression(("0.0"),degree=1)

	c = TrialFunction(Q)
	q = TestFunction(Q)

	#a = dot(grad(c), grad(q))/self.Pe*dx + dot(v, grad(c))*q*dx
	#L = f*q*dx
	F = dot(grad(c), grad(q))/self.Pe*dx + dot(v, grad(c))*q*dx - f*q*dx
        
	return lhs(F), rhs(F)
	#return a, L
