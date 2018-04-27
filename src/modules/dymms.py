""" This module contains functions and objects for solving time-dependent
Two-phase flow of a compacting matrix and an inviscid pore fluid"""
from dolfin import *
import numpy as np
import scipy, sys, math
import datetime

# Insert the definition of the symmetric gradient
def symgrad(u):
    """This function returns the symmetric gradient of
    a vector function u"""
    return(grad(u)+grad(u).T)

class domain:
    def __init__self(self,name):
        self.name=name
        self.da=[]
        self.R=[]
        self.B=[]
        self.theta=[]
        self.dL=[]

    def contiguity_whm12(self,phi):
        """ This function calculates the contiguity as a function
        of melt volume fraction phi. Works for up to phi = 0.25
        Wimert and Hier-Majumder 2012."""
        p1=-8065.00
        p2=6149.00
        p3=-1778.00
        p4=249.00
        p5=-19.77
        p6=1.00
        psi=p1*phi**5+p2*phi**4+p3*phi**3+p4*phi*phi+p5*phi+p6 

        return(psi)

    def vs_fwd(self,phi,psi):
        """ This function calculates dimensionless vs/v0
        as  a function of melt volume fraction and contiguity
        following Hier-Majumder et al 2014"""
        nu=0.25
        a1=1.8625+0.52594*nu-4.8397*nu**2
        a2=4.5001-6.1551*nu-4.3634*nu**2
        a3=-5.6512+6.9159*nu+29.595*nu**2-58.96*nu**3
        b1=1.6122+0.13527*nu
        b2=4.5869+3.6086*nu
        b3=-7.5395-4.8676*nu-4.3182*nu**2
        #Equation A5 of Takei 2001
        m=a1*psi+a2*(1.0-psi)+a3*psi*sqrt((1-psi)**3)
        #Equation A6
        n=b1*psi+b2*(1.0-psi)+b3*psi*(1-psi)**2
        #normalized bulk modulus of the skeletal framework eq 27, H-m 2008
        h=1.0-(1.0-psi)**m
        #normalized shear modulus of the skeletal framework eq 28, H-M 2008
        g=1.0-(1.0-psi)**n
        #Normalized shear modulus
        Novermu=(1.0-phi)*g
        #Ratio of fluid density over solid density
        rhofoverrho=1.0
        #average density normalized by solid density
        rhobaroverrho=1.0-phi+rhofoverrho*phi
        # Normalized shear wave speed
        VsoverVs0=sqrt(Novermu)/sqrt(rhobaroverrho)
        return(Novermu,VsoverVs0)

    def conductivity(self,cco2,ch2o,T,phi):
        """#Calculates the conductivity of CO2-H2O bearing
        #peridotite from Sifre etal. Nature, 2014
        Provide wt% of CO2, wt% of H2O concentration in
        melt and temperature in Kelvins, returns melt 
        conductivity in S/m"""

        #Parameters a,b,c,d,e for H2O from table T3
        a1=88774
        b1=0.388
        c1=73029
        d1=4.54e-5
        e1=5.5607
        #Parameters for CO2 from table 3
        a2=789166
        b2=0.1808
        c2=32820
        d2=5.5e-5
        e2=5.7956
        #Universal Gas constant
        R=8.314
        #Concentration of H2O and CO2 in wt%
        #The following values should be similar to 
        #the top curve in Figure 1
        #ch2o=10.
        #cco2=25.

        #Activation energies from eq 4
        EA1=a1*numpy.exp(-b1*ch2o)+c1
        EA2=a2*numpy.exp(-b2*cco2)+c2

        sig1=numpy.exp(d1*EA1+e1)
        sig2=numpy.exp(d2*EA2+e2)
        sigma1=sig1*numpy.exp(-EA1/R/T)
        sigma2=sig2*numpy.exp(-EA2/R/T)
        sigma=sigma1+sigma2

        # Now calculate the
        #bulk rock conductivity using Archie's law.
        #The parameters are from Yoshino et al. (2010), EPSL
        #Eq1 and Table 3
    
        C=0.67
        n=0.89
        sigbulk=C*phi**n*sigma
        return(sigbulk)

   
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
        equation in a multiphase  system. """
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
        variable density contrast and surface tension are supported"""

        U = TrialFunction(W)
        (v, q, omega) = TestFunctions(W)
        u, p, c   = split(U)
        
        one=Constant(1.0)
        two=Constant(2.0)
        three=Constant(3.0)
        four=Constant(4.0)
        alpha=two*(one-phi)*(two-phi)/3/phi
        eta=one/alpha
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

        h = (self.dL*(one-phi)*(chi*grad(phi)/self.B-self.R*buyoancy*zhat)\
             +self.da*grad(four*gam/three/phi))

        ###############################################

        F  = (one-phi)*inner(symgrad(u),symgrad(v))*dx\
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
## A separate class for Darcy flow of two phases
#######################################################
class darcy_two_phase:
    """This class contains the variables and functions
    for porous flow (no compaction) of a mixture of oil and 
    water"""
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
        of velocity equation. Also returns a preconditioner
        for the iterative solver"""
        U = TrialFunction(W)
        (v, omega) = TestFunctions(W)
        u, p   = split(U)
        ###############################################
        lam=1.0
        
        a  = (K*inner(u,v)/self.D/lam-p*div(v)\
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
   
    
    
#########################################################
def wedge_mesh(dimx=2,dimy=2,n=1):
    """This function uses a wedge shaped mesh from a box
    with lengths dimx and dimy, and degree of refinement n"""
    ######################################################
    ## First we use a mesh editor to create a slanted mesh
    editor = MeshEditor()
    init_mesh = Mesh()
    editor.open(init_mesh, dimx, dimy)  # top. and geom. dimension are both 2
    editor.init_vertices(6)  # number of vertices
    editor.init_cells(4)     # number of cells
    editor.add_vertex(0, np.array([1.01, 0.01]))
    editor.add_vertex(1, np.array([2.01, 0.01]))
    editor.add_vertex(2, np.array([0.01, 1.01]))
    editor.add_vertex(3, np.array([2.01, 1.01]))
    editor.add_vertex(4, np.array([1.01, 1.01]))
    editor.add_vertex(5, np.array([1.51, 0.01]))
    editor.add_cell(0, np.array([5, 1, 3], dtype=np.uintp))
    editor.add_cell(1, np.array([5, 4, 3], dtype=np.uintp))
    editor.add_cell(2, np.array([0, 5, 4], dtype=np.uintp))
    editor.add_cell(3, np.array([0, 4, 2], dtype=np.uintp))
    editor.close()
    print ('Dolfin Version',dolfin.dolfin_version())
    mesh=refine(init_mesh)
    # Refine the mesh n times
    
    for x in range(0, n):
        mesh=refine(mesh)
    return mesh
    ####################################################
