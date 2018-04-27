""" This module contains functions for solving mass and moemntum conservation
equations for single phase darcy flow, two-phase darcy flow, compaction, 
reactive darcy flow, and reactive compaction flow equations. Each class
contains the governing equations for the equations.

Detailed description of each class is provided within the classes. The solvers
and boundary conditions should be set within the script files calling on the
module functions. The multiphase physical properties are calculated by the 
module mumap_fwd. See details for these calculations in the docs for mumap_fwd.

Copyright Saswata Hier-Majumder, January 2018

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

        # Cut off any comments preceded by '#' from the string, a1 and a2 ignored  
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
    def __init__self(self):
        """ This initiates the compaction class. The following parameters
        are needed to initiate the class:
        Parameters
            da    :  Damkoehler number, nondimendional
            R     :  Density contrast, (rho_f-rho_s)/rho_s
            B     :  Bond number
            theta :  Dihedral angle at the melt-grain interface
            dL    :  ratio of delta/L, dimensionless
        These parameters are used by momentum and mass conservation equations.
        They can be read from a configuration file or supplied manually.
        """
        self.da=[]
        self.R=[]
        self.B=[]
        self.theta=[]
        self.dL=[]
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
    # Read in parameter file, store entries in dictionary
    def mass_conservation(self,V, phi0, u, dt, gam,mesh):
        """ This function solves for the mass conservation 
        equation in a multiphase  system. The governing PDE is
        described in the class description. The weak formulation
        is discussed in Appendix A of Allisic et al. (2014)
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
                                 -(1.0 - phi_mid)*div(u)) - self.da*gam)*dx
        # SUPG stabilisation term
        h_SUPG   = CellSize(mesh)
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
    def momentum_conservation(self,W, phi, gam,buyoancy):
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
        zhat=Constant((0.0, 0.0,1.0))
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
             -self.dL*phi*phi**m3 *inner(grad(p),grad(q))*dx \
             -omega*div(u)*dx - eta*c*omega*dx
        
        F -= -(inner(h, v)+self.da*self.R*buyoancy*gam*q\
              /(one-self.R*buyoancy))*dx
        # Bilinear form for the preconditioner     
        #b = (one-phi)*inner(symgrad(u),symgrad(v))*dx \
        #    +  self.dL*phi*phi*m3*inner(grad(p),grad(q))*dx\
        #    +alpha*p*q*dx + 0.5*alpha*c*omega*dx\
        #    +  self.dL*phi*phi*m3*inner(grad(p),grad(q))*dx\
        b = (one-phi)*inner(symgrad(u),symgrad(v))*dx+p*q*dx \
            + (1.0/alpha)*0.5*c*omega*dx
       
        return lhs(F), rhs(F), b
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
   
