""" This module contains functions for solving mass and moemntum conservation
equations for single phase darcy flow, two-phase darcy flow, compaction, 
reactive darcy flow, and reactive compaction flow equations. Each class
contains the governing equations for the equations.

Detailed description of each class is provided within the classes. The solvers
and boundary conditions should be set within the script files calling on the
module functions. The multiphase physical properties are calculated by the 
module mumap_fwd. See details for these calculations in the docs for mumap_fwd.

Copyright Saswata Hier-Majumder, January 2018
Modified by Joe Sun, February 2019
Works with Dolfin 2017.1.0
Python 2.7
"""
from dolfin import *
import numpy as np
import scipy, sys, math,os,string
import datetime

# Insert the definition of the symmetric gradient
def symgrad(u):
    """This function returns the symmetric gradient of
    a vector function u"""
    return(grad(u)+grad(u).T)
def iteration_output_info(niter,h,l2_u,l2_p,l2_c):
    """This function saves the output of iterations"""
    print '####################################'
    info("\n")
    info("Number of Krylov iterations: %g\n" %niter)
    info("Maximum Cell Diameter:  %g\n" %h)
    info("Velcoity solution: L2 norm:%e\n" % l2_u)
    info("Pressure solution: L2 norm:%e\n" % l2_p)
    info("Compaction solution: L2 norm:%e\n" % l2_c)
    print '####################################'
def iteration_output_write(niter,h,l2_u,l2_p,l2_c,fname="output/iter.out"):
    """This function saves the output of iterations"""
    file=open(fname,"a")
    file.write("####################################")
    file.write("\n")
    file.write("Number of Krylov iterations: %g\n" %niter)
    file.write("Maximum Cell Diameter:  %g\n" %h)
    file.write("Velcoity solution: L2 norm:%e\n" % l2_u)
    file.write("Pressure solution: L2 norm:%e\n" % l2_p)
    file.write("Compaction solution: L2 norm:%e\n" % l2_c)
    file.write("####################################")
    file.close    
########################################################
#### Utility functions for input
#### modified from Alisic et al. 2014
#######################################################

def int_float_string(val):
    """Convert string values to `correct' type if int or float"""

    # Integer?
    if val.isdigit(): 
        return int(val)

    # Float?
    try:
        return float(val)
    
    except ValueError:
    # Else assume string, return without any whitespace
        return val.replace(" ","")
# =========================================================
# Read in parameter file, store entries in dictionary
# Modified from Alisic et al. 2014
def parse_param_file(filename):
    """Parse a 'key = value' style control file, and return
    a dictionary of those settings"""

    settings = {}
    file     = open(filename, "r")

    # Read file into a list of lines
    try:
        lines = file.read().splitlines()
    finally:
        file.close()

    # Process each line
    for line in lines:

        # Skip blank lines
        if string.strip(line) == '':
            continue # to next line in control file

        # Skip comment lines starting with '#'
        if line.startswith('#'):
            continue # to next line in control file

        # Key/value pairs can be seperated by whitespaces
        opt = line.rstrip()
        opt = opt.replace(' ','') # remove any spaces padding

        # Cut off any comments preceded by '#' from the string,
        # a1 and a2 ignored  
        (opt, a1, a2) = opt.partition('#')    

        # Isolate the key and value
        (key, val)    = string.split(opt, '=')

        # Store in dictionary with correct data type
        # (i.e., integer, float, or string)
        val = int_float_string(val)

        #info('Parsing %s: %s = %s' % (filename, key, val))

        # Store as key val pair
        settings[key] = val
        
    return settings

    
#########################################
## end utility functions
##########################################

##########################################
## Classes for different problems
#########################################
class compaction:
    """ This class calculates the governing equations for a
    deforming, viscous matrix, and a pore fluid occupying interstitial
    spaces. The governing equations for momentum conservations
    are given by the two PDEs

    grad(alpha div(u))+div((1-phi)*symgrad(u))-grad(p)=h              (1)
    div(u - (delta/L)**2 * phi**2 * m *grad(p)) = Da *(R/(1-R))*Gamma (2)

    where u, the matrix velocity, and p,the modified fluid pressure,
    are the two primary unknowns. phi is the spatially and temporally variable
    melt volume fraction, delta is the compaction length, L 
    is the characteristic length of the problem, m is mobility, Da is 
    dimensionless Damkoehler number, R is the dimensionless density
    contrast between the melt and the matrix, alpha is bulk viscosity,
    and Gamma is the rate  of melt generation.

    The right hand side vector h comprises of buoyancy, surface tension,
    and melt generation.

    Mass conservation is given by the time dependent equation

    diff(phi,t) = div((1-phi)*u) + Gamma/rho                          (3)

    See the functions for these equations for weak formulation
    """
    def __init__(self,param_file,ksp="minres"):
        """ This initiates the compaction class. The following parameters
        are needed to initiate the class:
        Input    :
            param_file    :  The name of the file containing parameters.
                             This file name is passed from the command 
                             prompt. 
            ksp           :  Parameter for default Krylov solver 
        Assigned :
            da            :  Damkoehler number, nondimendional
            R             :  Density contrast, (rho_f-rho_s)/rho_s
            B             :  Bond number
            theta         :  Dihedral angle at the melt-grain interface
            dL            :  ratio of delta/L, dimensionless
            T             :  Total time for the simulation to run
            dt            :  Time step at each simulation
            out_freq      :  Frequency of writing output files
            krylov_method :  Krylov solver method for iterative
                             solution techniques. Default is minres
                             If minres is unavailable, then tfqmr
                             is used.
        These parameters are used by momentum and mass conservation equations.
        They can be read from a configuration file or supplied manually.
        During intiation
        """
        param         = parse_param_file(param_file)
        self.da       = param['da']
        self.R        = param['R']
        self.B        = param['B']
        self.theta    = param['theta']
        self.dL       = param['dL']
        self.T        = param['T']
        self.dt       = param['dt']
        self.out_freq = param['out_freq']
        # Test for PETSc or Tpetra
        if not has_linear_algebra_backend("PETSc") and \
           not has_linear_algebra_backend("Tpetra"):
            info("DOLFIN has not been configured with Trilinos\
            or PETSc. Exiting.")
            exit()

        if not has_krylov_solver_preconditioner("amg"):
            info("Sorry, this demo is only available when DOLFIN \
            is compiled with AMG " "preconditioner, Hypre or ML.")
            exit()

        if has_krylov_solver_method("minres"):
            self.krylov_method = ksp
        elif has_krylov_solver_method("tfqmr"):
            self.krylov_method = "tfqmr"
        else:
            info("Default linear algebra backend was not \
            compiled with MINRES or TFQMR "\
                 "Krylov subspace method. Terminating.")
            exit()
    def surface_tension_2(self,phi):
        """This function calculates the second derivative of surface tension
        with respect to phi. Uses the definition of contiguity from Wimert
        and hier-Majumder 2012"""
        p1=-8065.00
        p2=6149.00
        p3=-1778.00
        p4=249.00
        chi = (2.0*cos(self.theta)-1.0)*(2.0*p4+20.0*p1*phi**3+ \
                                     12.0*p2*phi**2+6.0*p3*phi)
        return(chi)
    def mass_conservation(self,V, phi0, u, dt, gam,mesh):
        """ 
        This function returns the bilinear form for mass conservation
        in a single-component two-phase system. The governing PDE is
        given by:

        diff(phi,t) = div((1-phi)*u) + Gamma/rho

        The weak formulation is discussed in Appendix A of
        Allisic et al. (2014). Outputs from this function
        can be used to call the function mass_solver, to solve
        the bilinear form using iterative sparse solvers.
        Input
            V      : Function space for melt volume fraction
            phi0   : Melt volume fraction from previous time step
            u      : Velocity from the current time step
            dt     : Length of current time step
            gam    : A function describing melt generation
            mesh   : Mesh for the problem
        Returns:
            lhs(F) : Left hand side of the bilinear form
            rhs(F) : Right hand side of the bilinear form
            b      : A preconditioner for iterative solution
        """
        phi1 = TrialFunction(V)
        w    = TestFunction(V)
        phi_mid = 0.5*(phi1+phi0)
        F = w*(phi1 - phi0 + dt*(dot(u, grad(phi_mid)) \
                                 -(1.0 - phi_mid)*div(u)) - dt*self.da*gam)*dx
        # SUPG stabilisation term
        #h_SUPG   = CellSize(mesh)
        h_SUPG   = CellDiameter(mesh)
        residual = phi1 - phi0 + dt * (dot(u, grad(phi_mid)) \
                                       - div(grad(phi_mid))-self.da*gam)
        unorm    = sqrt(dot(u, u))
        aval     = 0.5*h_SUPG*unorm
        keff     = 0.5*((aval - 1.0) + abs(aval - 1.0))
        stab     = (keff / (unorm * unorm)) * dot(u, grad(w)) * residual * dx
        F       += stab
        #Return a preconditioner for solving the time marching 
        #by iterative solution, if needed
        b=w*phi1*dx
        return lhs(F), rhs(F),b
    #########################################################
    def mass_solver(self,X,a_phi,L_phi,bb_phi,\
                    nits=100,tol=0.000001,monitor=False):
        """This function solves for the single component
        two-phase mass conservation equation derived in 
        the function mass conservation. The biinear form
        of the equation is
        a = L
        and the bilinear form of the preconditioner matrix
        is in b. The function mass_conservation must be
        called before calling this function to create the
        bilinear forms from the PDEs.
        Input:
            X       : Function space for storing the 
                      updated melt fraction at time step t_n+1
            a_phi   : bilinear form of LHS matrix
            L_phi   : bilinear form of RHS matrix
            bb_phi  : bilinear form of preconditioner matrix
        Parameters:
            nits    : Maximum iterations, for Krylov solver
            tol     : Relative tolerance for residuals of the
                      iterative solver
            monitor : A Boolean parameter for monitoring convergence
                      of the solution
        output:
            sol     : Solution containing the new melt fraction 
        """

        # Create a Krylov solver for mass conservation
        solver_phi = KrylovSolver(self.krylov_method, "amg")
        solver_phi.parameters["relative_tolerance"] = tol
        solver_phi.parameters["maximum_iterations"] = nits
        solver_phi.parameters["monitor_convergence"] = monitor

        # Assemble system for porosity advection
        A_phi, b_phi = assemble_system(a_phi, L_phi)
        # Associate a preconditioner with the porosity advection
        P_phi, btmp_phi=assemble_system(bb_phi, L_phi)
        #Connect operator to equations
        solver_phi.set_operators(A_phi, P_phi)
        # Solve
        sol=Function(X)
        
        solver_phi.solve(sol.vector(), b_phi)
        return sol

    def momentum_conservation(self,W, phi, gam,buyoancy,zh=Constant((0.0, 0.0,1.0))):
        """Return the bilinear form for momentum conservation
        equation using the split-field formulation, using velocity
        pressure and compaction as the three unknowns. Also returns
        a block-form preconditioner for iterative solutions. Spatially
        variable density contrast and surface tension are supported
        The combined weak form of the momentum conservation equations
        (1) and (2) are given by
        int((1-phi)*inner(symgrad(u),symgradv(v))-C*div(v)-p*div(v)
        -q*div(u) - phi**2*m*inner(gradp),grad(q))-omega*div(u)
        -eta*C*omega)*dx = int(-inner(h,v)+da*(R/(1-R))*Gamma*q)*dx
        
        where u,p, and C are trial functions and v,q, and omega are test
        functions. The weak formulation follows the split-field formulation
        by introducing a third PDE
        div(u) = C,
        which alleviates the saddle point problem.

        Input:
            W        : Function space containing both U, p, and c
            phi      : Melt volume fraction at the current time step
            gam      : A fucntion describing melt generation
            buyoancy : A function describing spatial variations in buyoancy
        Returns:
            lhs(F)   : Left hand side of the bilinear expression
            rhs(F)   : Right hand side of the bilinear expression
            b        : Bilinear form for preconditioner
        """

        U = TrialFunction(W)
        (v, q, omega) = TestFunctions(W)
        u, p, c   = split(U)
        
        one=Constant(1.0)
        two=Constant(2.0)
        three=Constant(3.0)
        four=Constant(4.0)
        alpha=2.0*(1.0-phi)*(2.0-phi)/3.0/phi
        eta=1.0/alpha
        #Two forms of mobility dependence on melt volume fraction
        #Tubes
        m2 =1.0/phi/phi
        #Films
        m3=1.0/(phi*phi*phi)
        zhat=zh
        ###############################################
        #This is the new bit about surface tension, 
        # melting and density contrast
        chi=self.surface_tension_2(phi)

        h = self.dL*(one-phi)*(chi*grad(phi)/self.B-self.R*buyoancy*zhat)\
             +self.da*grad(four*gam/three/phi)

        ###############################################

        F  = 0.5*(one-phi)*inner(symgrad(u),symgrad(v))*dx\
             - p*div(v)*dx \
             - c*div(v)*dx \
             - q*div(u)*dx \
             +self.dL*phi*phi**m3 *inner(grad(p),grad(q))*dx \
             -omega*div(u)*dx - eta*c*omega*dx
        #F  = (one-phi)*inner(grad(u),grad(v))*dx\
        #     - p*div(v)*dx \
        #     - c*div(v)*dx \
        #     - q*div(u)*dx \
        #     -self.dL*phi*phi**m3 *inner(grad(p),grad(q))*dx \
        #     -omega*div(u)*dx - eta*c*omega*dx
        
        L= -(inner(h, v)+self.da*self.R*buyoancy*gam*q\
              /(one-self.R*buyoancy))*dx
        # Bilinear form for the preconditioner     
        b = 0.5*(one-phi)*inner(symgrad(u),symgrad(v))*dx+p*q*dx \
            + (1.0/alpha)*0.5*c*omega*dx
        b = (one-phi)*inner(grad(u), grad(v))*dx \
            +  self.dL*phi*phi*m3*inner(grad(p),grad(q))*dx + p*q*dx+ 0.5*eta*c*omega*dx
        #
        return F, L, b
    def momentum_solver(self, W, a , L, b, bcs,pc='amg',\
                        tol=0.000001,max_its=3000,  monitor=False):
        """This function sets up the amg krylov
        solver for the momentum conservation equation
        a = L 
        Input paramaters are
        W       =  Mixed function space for velocity, pressure
                   and compaction
        a       =  bilinear left hand side containing u,p,C
        L       =  bilinear right hand side containing knowns
        b       =  preconditioner generated by momentum conservation
        bcs     =  Boundary conditions
        
        tol     =  Relative tolerance for KSP solver convergence
        max_its =  Maximum KSP iterations
        monitor =  Logical parameter for monitoring convergence
                   of Krylov solution
        Output
        u       = Function containing velocity vector
        p       = Function containing pressure
        c       = Function containing compaction
        niter   = integer number of ksp iterations required to 
                  reach convergence
        """
       

        # Create Krylov solver and AMG preconditioner
        solver = KrylovSolver(self.krylov_method, "amg")
        solver.parameters["relative_tolerance"] = tol
        solver.parameters["maximum_iterations"] = max_its
        solver.parameters["monitor_convergence"] = monitor

        A_compact, b_compact = assemble_system(a, L, bcs)
        # Assemble preconditioner system
        P, btmp = assemble_system(b, L, bcs)
        # Associate operator (A) and preconditioner matrix (P)
        solver.set_operators(A_compact, P)

        # Solve
        sol=Function(W)
        #niter=solve(a==L,sol,bcs,form_compiler_parameters={"quadrature_degree": 3, "optimize": True})
        
        niter=solver.solve(sol.vector(),b_compact)
        # Get sub-functions
        u, p, c = sol.split()
        
        return u,p,c,niter
    def momentum_solver_direct(self, W, a , L, b, bcs,pc='amg',\
                        tol=0.000001,max_its=3000,  monitor=False):
        """This function sets up the amg krylov
        solver for the momentum conservation equation
        a = L 
        Input paramaters are
        W       =  Mixed function space for velocity, pressure
                   and compaction
        a       =  bilinear left hand side containing u,p,C
        L       =  bilinear right hand side containing knowns
        b       =  preconditioner generated by momentum conservation
        bcs     =  Boundary conditions
        
        tol     =  Relative tolerance for KSP solver convergence
        max_its =  Maximum KSP iterations
        monitor =  Logical parameter for monitoring convergence
                   of Krylov solution
        Output
        u       = Function containing velocity vector
        p       = Function containing pressure
        c       = Function containing compaction
        niter   = integer number of ksp iterations required to 
                  reach convergence
        """
       

        # Create Krylov solver and AMG preconditioner
        solver = KrylovSolver(self.krylov_method, "amg")
        solver.parameters["relative_tolerance"] = tol
        solver.parameters["maximum_iterations"] = max_its
        solver.parameters["monitor_convergence"] = monitor

        
        # Solve
        sol=Function(W)
        niter=solve(a==L,sol,bcs,form_compiler_parameters={"quadrature_degree": 3, "optimize": True})
        
        #niter=solver.solve(sol.vector(),b_compact)
        # Get sub-functions
        u, p, c = sol.split()
        
        return u,p,c,niter

########################################################
## A class for Darcy flow of two phases
#######################################################
class darcy_two_phase:
    """This class contains the variables and functions
    for porous flow of a mixture of fluids. See Cheueh et al (2010)
    for details of the governing equation and weak formulation.
    The governing PDEs are given by
    phi*diff(S,t) + div(F*u) = qw      (1)
    div(u) = q                         (2) 
    u = -D*lambda*k*grad(p)            (3)
    where phi is the porposity, S is saturation, u is the velocity,
    and q_w is source term for one of the fluid phases. The 
    dimensionless quantity F is the ratio between mobility of the
    fluid over the total mobility. 
    q =qw+qo total source function
    D is the diemnsionless Darcy number
    lambda is the dimensionless mobility
    and p is the modified pressure.
    """
    def __init__(self,D=1.0):
        """Initiate the class with Darcy number"""
        self.D = D
    def mobility(self, S):
        """Calculates the total mobility for a given
        water saturation"""
        sol=S**2
        return sol
    def fractional_flow(self,S):
        """This function calculates the fractional flow
        of water in the pore space"""
        sol=S**2
        return sol
    def darcy_solver(self,W,mesh,K,q):
        """Solves the Darcy flow equation and the divergence
        of velocity equation. The weak formulation is givenn by
        int((1/D/lambda)*inverse(K)*inner(u,v)-p*div(v)
        -p*div(v)+div(u)*omega )*dx = int(p*inner(v,n)*dS
        +int(q*omega)*dx
        On input:
            W    : Function space
            mesh : Current mesh
            K    : Permeability tensor/scalar
            q    : Function for source
        Returns
            a    : Right hand side of bilinear form
            L    : Left hand side of bilinear form
            b    : Preconditioner for iterative solver                   
        """
        U = TrialFunction(W)
        (v, omega) = TestFunctions(W)
        u, p   = split(U)
        ###############################################
        lam=1.0
        
        a  = (inner(u,v)/self.D/lam/K-p*div(v)\
        +div(u)*omega)*dx
        norm = FacetNormal(mesh)
        L = -q*omega*dx 
        # Bilinear form for the preconditioner     
        b = (K*inner(u,v)/self.D/lam-div(u)*omega\
             -p*div(v)- K*inner(u,v)/self.D/lam)*dx
        #Define a goal functional
        return a,L,b
    def mass_conservation(self,V, S0, u, dt, phi, F, qw,mesh):
        """ This function solves for the mass conservation 
        equation in a multiphase  system. """
        S1 = TrialFunction(V)
        w  = TestFunction(V)
        S_mid = 0.5*(S1+S0)
        F = w*((S1 - S0)*phi + dt*(F*div(u) - qw))*dx
        # SUPG stabilisation term
        #h_SUPG   = CellSize(mesh)
        #residual = w*((S1 - S0)*phi + dt*(F*div(u) - qw))
        #unorm    = sqrt(dot(u, u))
        #aval     = 0.5*h_SUPG*unorm
        #keff     = 0.5*((aval - 1.0) + abs(aval - 1.0))
        #stab     = (keff / (unorm * unorm)) * dot(u, grad(w)) * residual * dx
        stab    = 0.0
        F       += stab
        #Return a preconditioner for solving the time marching 
        #by iterative solution, if needed
        b=w*S1*dx
        return lhs(F), rhs(F),b
###################################################################
### Advection diffusion equation in Darcy flow
###################################################################
class DarcyAdvection():
    """
    This class solves for a simple advection-diffusion
    equation for a single or multicomponent flow, the governing
    PDEs for Darcy flow are:        
    div(u) = 0                                                          (1)
    phi*u = -k*(grad(p)-drho*zhat)                                      (2)
    dc0/dt + dot(u,grad(c0)) = div(grad(c0))/Pe - Da*c0*c1/phi + beta*f (3)
    and  
    dc1/dt = - Da*c0*c1/phi                                             (4)
    where c0 and c1 are concentrations of the reactants in the liquid
    and solid, u is the fluid velocity, p is pressure, k is permeability
    drho=difference between liquid and solid densities, zhat is a unit
    vecotr in vertically upward direction, Pe is Peclet number, Da is
    the Dahmkoler number, beta is source strength, f is a function
    for lateral variations in source of c0, and phi is the constant porosity

    On initiation of the class:
              the dimensionless numbers, Pe, Da, alpha =beta*phi are loaded
              the timestep dt and CFL criterion are also loaded to default
              values.
    For the remaining functions, see the docstring of each individual function
    for help.
    """
    
    def __init__(self,Pe=100,Da=10.0,phi=0.01,alpha=0.005,cfl=1.0e-2,dt=1.0e-2):
        """Initiates the class. Inherits nondimensional numbers
        from compaction, only need to add Pe"""
        self.Pe=Pe
        self.Da=Da
        self.phi=phi
        self.cfl=cfl
        self.dt=dt
        self.alpha=alpha
        
    def darcy_bilinear(self,W,mesh,K=0.1,zh=Constant((0.0,1.0)),TwodTrue=True):
        """
        This function creates the bilinear form of the Darcy flow equations:
        div(u) = 0                            (1)
        phi*u = -k*(grad(p)-drho*zhat)        (2)
        can be used for a stand-alone Darcy flow formulation without concentration
        profiles. 
        Input:
             W      : Mixed function space for velocity and pressure
             mesh   : Fenics mesh on which W is defined
             K      : Constant permeability
             zhat   : 2D vertical unit vector
             TwodTrue : True if the problem is 2D
        Output:
             a      : Left hand side of the weak formulation a=L
             L      : Right hand side of the weak formulation
        If a lateral variation in permeability exists, it can be incorporated
        through the function f
        """
        U = TrialFunction(W)
        (v, q) = TestFunctions(W)
        u, p   = split(U)
        if TwodTrue:
            zhat=zh        
        else:
            zhat=Constant((0.0,0.0,1.0))
        f=Expression("1.0",degree=1)
        # Define the variational form
        a = (inner(u,v)-K*div(v)*p+div(u)*q)*dx
        L = K*f*inner(v,zhat)*dx
        return(a,L)
    
    def advection_diffusion(self,Q, u0, velocity, dt,mesh):
        """
        This function creates the bilienar form for advection reaction equations
        dc0/dt + dot(u,grad(c0)) = div(grad(c0))/Pe - Da*c0 (3)
                                  (4)
        This function requires the knowledge of the velocity vector for advection.
        Can be used as a stand-alone diffusion-advection solver if the velocity is 
        already known.
        Input:
             Q       : Scalar function space for two concentration fields
             u0      : Concentration field from the last timestep
             velocity: Known vector field/function 
             dt      : Time step for iteration
             mesh    : Fenics mesh on which Q is defined
        Output:
             lhs(F)  : Left hand side of the bilinear form a=L
             rhs(F)  : Right hand side of the bilinear form
        This uses a Crank-Nicholson discretization of the time stepping. This is
        incorporated in the term u_mid. The first order reaction rate is in f. It also
        uses a simplified SUPG stabilization term using the cell diameter and norm
        of the velocity vector.       
        """
        
        h = CellDiameter(mesh)
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
    

    def advection_diffusion_two_component_adaptive(self,Q, c_prev, velocity,mesh):
        """ This function builds the bilinear form for component advection
        diffusion for component one. The source term depends on the concentration
        of both components. Concentration of the other component should be
        calculated outside this function after solving the bilinear form from
        this step"""
        
        f1  = Expression("0.005*(1.0-tanh(x[1]/0.2))*sin(2.0*x[0]*3.14)",degree=1)
        f2  = Expression("0.001",degree=1)
        h = CellDiameter(mesh)
        # Parameters
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
        # the source term for c0 f1=[0,1]
	f1 = Expression('(1.0-tanh(x[1]/0.01))*( (1.0+sin(1.0*x[0]*3.14)) \
		+ (1.0+sin(2.0*x[0]*3.14)) + (1.0+sin(3.0*x[0]*3.14)) \
		+ (1.0+sin(4.0*x[0]*3.14)) + (1.0+sin(5.0*x[0]*3.14)) \
		+ (1.0+sin(6.0*x[0]*3.14)) + (1.0+sin(7.0*x[0]*3.14)) \
		+ (1.0+sin(8.0*x[0]*3.14)) + (1.0+sin(9.0*x[0]*3.14)) \
		+ (1.0+sin(10.0*x[0]*3.14)) )/20.0',degree=1)
        ###############################################
        # Adaptive time-stepping added September 2018
        # Right hand side of advection equation
        # u.grad(c)-div(grad(c))/Pe+f
        # Evaluate this term from the last time step
        advect_term=dot(velocity, grad(u0)) - div(grad(u0))/self.Pe+f
        # Create a DG function sapce to evaluate the values of this
        DG = FunctionSpace(mesh, "DG", 0)
        #Save it into a function 
        advect_rhs=project(advect_term,DG)
        # Now evaluate the maximum value of this term
        ad_max=advect_rhs.vector().max()
        #Compute dt for this time step such that
        #dt*ad_max<=CFL
        #if np.abs(ad_max)>self.cfl:
        #    self.dt=self.cfl/ad_max      
        dt=self.dt
        #####End adaptive time stepping
        ##################################################
        
        # Galerkin variational problem
        #F = v*(u - u0)*dx + dt*(v*dot(velocity, grad(u_mid))*dx\
        #        + dot(grad(v), grad(u_mid)/self.Pe)*dx)+(dt*f*v/self.phi)*dx\
        #        + q*(c - c0)*dx + dt*f*q*dx
        # Residual
        #r = u - u0 + dt*(dot(velocity, grad(u_mid)) - div(grad(u_mid))/self.Pe+self.Da*u0*c0)\
        #    +c-c0+dt*self.Da*u0*c0/self.phi
        r = u - u0 + dt*(dot(velocity, grad(u_mid)) - div(grad(u_mid))/self.Pe+f/self.phi)\
            +c-c0+dt*f/(1.0-self.phi)
        # Add SUPG stabilisation terms
        vnorm = sqrt(dot(velocity, velocity))
        #F += (h/(2.0*vnorm))*dot(velocity, grad(v))*r*dx

        alpha_SUPG=self.Pe*vnorm*h/2.0
        #Brookes and Hughes
        coth = (np.e**(2.0*alpha_SUPG)+1.0)/(np.e**(2.0*alpha_SUPG)-1.0)
        term1_SUPG=0.5*h*(coth-1.0/alpha_SUPG)/vnorm
        #Sendur 2018
        tau_SUPG = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h)
        #Codina 1997 eq. 114
        #tau_SUPG = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h+self.Da)
        tau_SUPG = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h+self.Da*np.max(c0)/self.phi)
        term_SUPG = tau_SUPG*dot(velocity, grad(v))*r*dx

        F = v*(u - u0)*dx + dt*(v*dot(velocity, grad(u_mid))*dx \
                                + dot(grad(v), grad(u_mid)/self.Pe)*dx) \
                                + dt*f/self.phi*v*dx  - self.alpha/self.phi*dt*f1*v*dx\
                                + q*(c - c0)*dx + dt*f/(1-self.phi)*q*dx
        
        F += term_SUPG
        return lhs(F), rhs(F)
    
    def advection_diffusion_two_component(self,W,mesh,sol_prev,dt,f1,K=0.1,zh=Constant((0.0,1.0)),SUPG=1):
        """     
        This function returns the bilinear form for the system of PDES governing
        an advection-reaction-two-component flow, the governing
        PDEs for Darcy flow are:        
        div(u) = 0                                                          (1)
        phi*u = -k*(grad(p)-drho*zhat)                                      (2)
        dc0/dt + dot(u,grad(c0)) = div(grad(c0))/Pe - Da*c0*c1/phi + beta*f (3)
        and  
        dc1/dt = - Da*c0*c1/phi                                             (4)
        where 
        drho = 1 + c0                                                       (5)
        where c0 and c1 are concentrations of the reactants in the liquid
        and solid, u is the fluid velocity, p is pressure, k is permeability
        drho=difference between liquid and solid densities, zhat is a unit
        vecotr in vertically upward direction, Pe is Peclet number, Da is
        the Dahmkoler number, beta is source strength, f is a function
        for lateral variations in source of c0, and phi is the constant porosity
        Input:
             W        : Mixed function space containing velocity, pressure and two
                        concentrations
             mesh     : Fenics mesh on which W is defined
             sol_prev : All unknowns from the previous time step
             dt       : time step
             f1       : Spatially variable scalar function for the source term
             K        : Constant permeability
             zh       : Unit vector in the vertical direction
        Output:
             lhs(F)   : Left hand side of the combined bilinear form
             rhs(F)   : Right hand side of the bilinear formulation
        The bilinear form uses Crank-Nicholson time stepping and gives 
        several options for SUPG stabilization. We recommend using Sendur 2018 formulation
        for the SUPG term.
        The bilinear form, combined together, becomes
        phi*dot(u,v)-k*p*div(v)+div(u)*q + eta*(c0n-c0nm1)+eta*dot(u,grad(c_mid))*dt
                 +dt*dot(grad(c_mid),grad(eta))/Pe+(c1_n-c1_nm1)*omega
                 = K*drho*dot(zhat,v)-Da*c0*c1*dt*eta/phi 
                   - beta*f1*dt*eta + Da*c0*c1*dt*omega/(1-phi)
        In this code c0 is uc, c1 is cc, eta is vc, omega is qc
        We recommend using this function preferentially over the other functions as
        this combines all four equations into one bilinear form. If the density contrast
        is constant, just replace drho with a constant value.
        """
	h = CellDiameter(mesh)
	# TrialFunctions and TestFunctions
        U = TrialFunction(W)
        (v, q, vc, qc) = TestFunctions(W)
        u, p, uc, cc   = split(U)
        zhat=zh
        deltarho=(1.0+uc)
        # Define the variational form
        F = (inner(self.phi*u,v) - K*div(v)*p+div(u)*q)*dx - K*deltarho*inner(v,zhat)*dx
        # uc and cc are the trial functions for the next time step
        # uc for comp cc and d comp1 
        # u0 (component 0) and c0(component 1) are known values from the previous time step
        u,p,u0 ,c0 = split(sol_prev)
        # Mid-point solution for comp 0
        u_mid = 0.5*(u0 + uc)
        # First order reaction term
        f = self.Da*u0*c0
	F += vc*(uc - u0)*dx + dt*(vc*dot(u, grad(u_mid))*dx\
                + dot(grad(vc), grad(u_mid)/self.Pe)*dx) \
		+ dt*f/self.phi*vc*dx  - self.alpha/self.phi*dt*f1*vc*dx \
                + qc*(cc - c0)*dx + dt*f/(1-self.phi)*qc*dx
        # Residual
        h = CellDiameter(mesh)
        r = uc - u0 + dt*(dot(u, grad(u_mid)) - div(grad(u_mid))/self.Pe+f/self.phi)\
            - self.alpha/self.phi*dt*f1 + cc-c0 + dt*f/(1.0-self.phi)
        # Add SUPG stabilisation terms
        # Default is Sendur 2018, a modification of Codina, 1997 also works
        vnorm = sqrt(dot(u, u))
        
        if SUPG==1:
            #Sendur 2018
            tau_SUPG = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h)
        elif SUPG==2:
            #Codina 1997 eq. 114
            tau_SUPG = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h+self.Da)
            #tau_SUPG = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h+self.Da*np.max(c0)/self.phi)
        else:
            alpha_SUPG=self.Pe*vnorm*h/2.0
            #Brookes and Hughes
            coth = (np.e**(2.0*alpha_SUPG)+1.0)/(np.e**(2.0*alpha_SUPG)-1.0)
            tau_SUPG=0.5*h*(coth-1.0/alpha_SUPG)/vnorm
        term_SUPG = tau_SUPG*dot(u, grad(vc))*r*dx
        F += term_SUPG       
        return lhs(F), rhs(F)
    
###################################################################
### Advection diffusion equation in Stokes flow
###################################################################
class StokesAdvection():
    """
    This class solves for a simple advection-diffusion
    equation for a single or multicomponent flow, the governing
    PDEs for Stokes flow are:        
    div(u) = 0                                                          (1)
    -grad(p)+mu*div(grad(u)) +drho*g =0                                 (2)
    dc0/dt + dot(u,grad(c0)) = div(grad(c0))/Pe - Da*c0*c1/phi + beta*f (3)
    and  
    dc1/dt = - Da*c0*c1/phi                                             (4)
    where c0 and c1 are concentrations of the reactants in the liquid
    and solid, u is the fluid velocity, p is pressure, k is permeability
    drho=difference between liquid and solid densities, zhat is a unit
    vecotr in vertically upward direction, Pe is Peclet number, Da is
    the Dahmkoler number, beta is source strength, f is a function
    for lateral variations in source of c0, and phi is the constant porosity

    On initiation of the class:
              the dimensionless numbers, Pe, Da, alpha =beta*phi are loaded
              the timestep dt and CFL criterion are also loaded to default
              values.
    For the remaining functions, see the docstring of each individual function
    for help.
    """
    
    def __init__(self,Pe=100,Da=10.0,alpha=0.005,cfl=1.0e-2,dt=1.0e-2):
        """Initiates the class. Inherits nondimensional numbers
        from compaction, only need to add Pe"""
        self.Pe=Pe
        self.Da=Da
        self.cfl=cfl
        self.dt=dt
        self.alpha=alpha
    def stokes(self,W,mesh,sol_prev,dt,f1,K=0.1,zh=Constant((1.0,0.0,0.0))):
        """ This function substitutes in stokes flow instead of darcy flow
            for use in pore space flow modelling where the solid phase has been
            removed, making a phi value redundant            
        """
	h = CellDiameter(mesh)

	# TrialFunctions and TestFunctions
        U = TrialFunction(W)
        (v, q, vc, qc) = TestFunctions(W)
        u, p, uc, cc   = split(U)
        
        zhat=zh

        # Define the variational form
	F = inner(grad(u),grad(v))*dx + div(v)*p*dx + q*div(u)*dx - inner(zhat, v)*dx
        # Velocity is constant unless dependent on density as per
        # the darcy_advection_rho_posi_random function 		through the 1.0+uc term 

        # uc and cc are the trial functions for the next time step
        # uc for comp cc and d comp1 
        # u0 (component 0) and c0(component 1) are known values from the previous time step
        u,p,u0 ,c0 = split(sol_prev)

        # Mid-point solution for comp 0
        u_mid = 0.5*(u0 + uc)

        # First order reaction term
        f = self.Da*u0*c0

	F += vc*(uc - u0)*dx + dt*(vc*dot(u, grad(u_mid))*dx\
                + dot(grad(vc), grad(u_mid)/self.Pe)*dx) \
		+ dt*f*vc*dx  - self.alpha*dt*f1*vc*dx \
                + qc*(cc - c0)*dx + dt*f*qc*dx
        # Residual
        h = CellDiameter(mesh)
        r = uc - u0 + dt*(dot(u, grad(u_mid)) - div(grad(u_mid))/self.Pe+f)\
            - self.alpha*dt*f1 + cc-c0 + dt*f
        # Add SUPG stabilisation terms
        vnorm = sqrt(dot(u, u))
        

        #alpha_SUPG=self.Pe*vnorm*h/2.0
        #Brookes and Hughes
        #coth = (np.e**(2.0*alpha_SUPG)+1.0)/(np.e**(2.0*alpha_SUPG)-1.0)
        #term1_SUPG=0.5*h*(coth-1.0/alpha_SUPG)/vnorm
        #Sendur 2018
        #####tau_SUPG = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h)
        #Codina 1997 eq. 114
        tau_SUPG = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h+self.Da)
        #tau_SUPG = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h+self.Da*np.max(c0)/self.phi)
        term_SUPG = tau_SUPG*dot(u, grad(vc))*r*dx
        F += term_SUPG
        
        return lhs(F), rhs(F)

    def stokes_no_alpha(self,W,mesh,sol_prev,dt,K=0.1,zh=Constant((1.0,0.0,0.0))):
        """ This function substitutes in stokes flow instead of darcy flow
            for use in pore space flow modelling where the solid phase has been
            removed, making a phi value redundant            
        """
	h = CellDiameter(mesh)

	# TrialFunctions and TestFunctions
        U = TrialFunction(W)
        (v, q, vc, qc, qc1) = TestFunctions(W)
        u, p, uc, cc, cc2   = split(U)
        
        zhat=zh

        # Define the variational form
	F = inner(grad(u),grad(v))*dx + div(v)*p*dx + q*div(u)*dx - inner(zhat, v)*dx
        # Velocity is constant unless dependent on density as per the darcy_advection_rho_posi_random 		function through the 1.0+uc term 

        # uc and cc are the trial functions for the next time step
        # uc for comp cc and d comp1 
        # u0 (component 0) and c0(component 1) are known values from the previous time step
        u,p,u0 ,c0,cc0 = split(sol_prev)

        # Mid-point solution for comp 0
        u_mid = 0.5*(u0 + uc)

        # First order reaction term
        f = self.Da*u0*c0

	F += vc*(uc - u0)*dx + dt*(vc*dot(u, grad(u_mid))*dx\
                + dot(grad(vc), grad(u_mid)/self.Pe)*dx) \
		+ dt*f*vc*dx\
                + qc*(cc - c0)*dx + dt*f*qc*dx + qc1*(cc2 - cc0)*dx - dt*f*qc1*dx
	# Positive dt*f*qc*dx for consumption of Ca
	# Negative dt*f*qc1*dx for precipitation of C

        # Residuals
        h = CellDiameter(mesh)
        r = uc - u0 + dt*(dot(u, grad(u_mid)) - div(grad(u_mid))/self.Pe+f)\
            + cc-c0 + dt*f + cc2 - cc0 -dt*f
        # Add SUPG stabilisation terms
        vnorm = sqrt(dot(u, u))
       


        #alpha_SUPG=self.Pe*vnorm*h/2.0
        #Brookes and Hughes
        #coth = (np.e**(2.0*alpha_SUPG)+1.0)/(np.e**(2.0*alpha_SUPG)-1.0)
        #tau_SUPG=0.5*h*(coth-1.0/alpha_SUPG)/vnorm
        #Sendur 2018
        #tau_SUPG = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h)
        #Codina 1997 eq. 114
        #tau_SUPG = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h+self.Da)
        tau_SUPG = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h+self.Da)
        #tau_SUPG = 1.0/(4.0/(self.Pe*h*h)+2.0*vnorm/h+self.Da*np.max(c0)/self.phi)
        term_SUPG = tau_SUPG*dot(u, grad(vc))*r*dx
        F += term_SUPG
        
        return lhs(F), rhs(F)
