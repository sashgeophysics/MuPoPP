#######################################
#######################################
## Copyright Saswata Hier-Majumder, 2017
########################################
# This module contains functions calculating
# analytical solutions for two-phase flow
# in a sphere and a 2D wedge.
# # The functions in this module require Dolfin, mshr, numpy, and scipy
# libraries of python. 
########################################
from dolfin import*
from mshr import*
import numpy as np
from scipy.special import spherical_in as spin
from scipy.special import spherical_kn as spkn
from scipy import stats,special
import matplotlib as mp
def k3(x):
    """k3"""
    term = np.exp(-x)*(x**3+6.0*x**2+15.0*x+15.0)/x**4
    return term
def k2(x):
    """k2"""
    term = np.exp(-x)*(x**2+3.0*x+3.0)/x**3
    return term
def i3(x):
    """i3"""
    term = ((x**3+15.0*x)*np.cosh(x)-(6.0*x**2+15.0)*np.sinh(x))/x**4
    return term
def i2(x):
    """i2"""
    term = ((x**2+3.0)*np.sinh(x)-3.0*x*np.cosh(x))/x**3
    return term
class wedge_flow:
    """This class contains functions for analytical solutions
    for 2-phase flow in a wedge. The soultions are from Spiegelman
    and McKenzie(1987). We modify coefficients C and D by u0.
    Also calculates the residual in velocity"""
    def __init__(self,phi0,u0):
        """Initiates the class by setting the coefficients"""
        self.phi0=phi0
        self.u0=u0
    def calculate_wedge_flow(self,mesh):
        """This function calculates the 2D velocity field
        inside a wedge, driven by a prescribed velocity
        at the slanted face"""
        U= FunctionSpace(mesh, "CG", 2)
        # Write the pressure field
        P=Function(U)
        u=wedge_pressure(phi0=self.phi0,u0=self.u0,element=U.ufl_element())
        P.interpolate(u)
        #Velocity field        
        V = VectorFunctionSpace(mesh, "CG", 2)
        vel=Function(V)
        w=wedge_velocity(phi0=self.phi0,u0=self.u0,element=V.ufl_element())
        vel.interpolate(w)
        
        return vel,P

### Inherited expressions for wedge pressure and velocity solutions        
class wedge_pressure(Expression):
    """Calculates pressure within the wedge"""
    def __init__(self,phi0,u0,element):
        """Initiates the class by setting the coefficients"""
        self.phi0=phi0
        self.u0=u0
        #Turcotte and Schubert assumes u=v=u0/sqrt(2) at the slab surface
        self.CC=4.0*u0*pi/(pi**2 -8.0)
        self.BB=-self.CC #from top surface BC, u=0,v=0 @y=0
        self.DD=4.0*self.u0*(pi-4.0)/(pi**2-8.0)
    def eval(self,value,x):
        z1=x[1]-1.0
        x1=x[0]
        r2=x1**2+z1**2
        value[0]=-2.0*(1.0-self.phi0)*(self.CC*x1-self.DD*z1)/r2
        
class wedge_velocity(Expression):
    """Calculates velocity within the wedge"""
    def __init__(self,phi0,u0,element):
        """Initiates the class by setting the coefficients"""
        self.phi0=phi0
        self.u0=u0
        #Turcotte and Schubert assumes u=v=u0/sqrt(2) at the slab surface
        self.CC=4.0*u0*pi/(pi**2 -8.0)
        self.BB=-self.CC #from top surface BC, u=0,v=0 @y=0
        self.DD=4.0*self.u0*(pi-4.0)/(pi**2-8.0)
    def eval(self,value,x):
        z1=x[1]-1.0
        x1=x[0]
        r2=x1**2+z1**2
        value[0]=-self.BB - self.DD*np.arctan(z1/x1)\
            -x1*(self.CC*x1+self.DD*z1)/r2
        value[1]=self.CC*np.arctan(z1/x1)\
                  -z1*(self.CC*x1+self.DD*z1)/r2        
       
       
 # A class for spherical flow problems       
    
class spherical_flow:
    """This class contains the functions for flow in and around 
    a sphere. The solutions are calculated using Pakovitch-Neuber potential
    See Hier-Majumder (2017), JFM article for details
    The functions in this module require Dolfin, mshr, numpy, and scipy
    libraries of python. The solutions
    are ouputted in vtk files"""
    def __init__(self,a1,l1,b1,E):
        """initiate the class with the following variables
        a1= radius of sphere
        b1=ratio between bulk and shear viscosity
        l1=ratio between shear viscosities of the sphere and the
        substrate around the shear"""
        self.a1=a1
        self.b1=b1
        self.l1=l1
        self.E = E
    def calculate_internal_flow(self,innermesh):
        """Calculates velocity, pressure, compaction 
        and traction inside the sphere"""
        V = VectorFunctionSpace(innermesh, "CG", 2)
        U= FunctionSpace(innermesh, "CG", 2)
        # Internal flow
        # Write the compaction field for internal flow in pure shear
        fname="output/spherical_compaction_internal_lambda"+str(self.l1)\
            +"beta"+str(self.b1)+".pvd"
        C=Function(U)
        u=spherical_compaction_internal(a=self.a1,b=self.b1,l=self.l1, \
                              strain=self.E,element=U.ufl_element())
        C.interpolate(u)
        C_out=File(fname)
        C_out <<C
        # Write the pressure field for internal flow in pure shear
        fname="output/spherical_pressure_internal_lambda"+str(self.l1)\
            +"beta"+str(self.b1)+".pvd"
        P=Function(U)
        u=spherical_pressure_internal(a=self.a1,b=self.b1,l=self.l1\
                            ,strain=self.E,element=U.ufl_element())
        P.interpolate(u)
        C_out=File(fname)
        C_out <<P
        # Write the pressure difference at the boundary
        fname="output/dP_boundary_lambda"+str(self.l1)\
            +"beta"+str(self.b1)+".pvd"
        P=Function(U)
        u=dP_boundary(a=self.a1,b=self.b1,l=self.l1\
                      ,strain=self.E,element=U.ufl_element())
        P.interpolate(u)
        C_out=File(fname)
        C_out <<P
        
        # Write the normal gradient of pressure/melt flux
        fname="output/spherical_flux_internal_lambda"+str(self.l1)\
            +"beta"+str(self.b1)+".pvd"
        P=Function(U)
        u=spherical_flux_internal(a=self.a1,b=self.b1,l=self.l1\
                        ,strain=self.E,element=U.ufl_element())
        P.interpolate(u)
        C_out=File(fname)
        C_out <<P
        #Velocity field
        V = VectorFunctionSpace(innermesh, "CG", 2)
        fname="output/spherical_velocity_internal_lambda"+str(self.l1)\
            +"beta"+str(self.b1)+".pvd"
        vel=Function(V)
        w=spherical_velocity_internal(a=self.a1,b=self.b1,l=self.l1\
                            ,strain=self.E,element=V.ufl_element())
        vel.interpolate(w)
        v_out=File(fname)
        v_out<<vel

        #Traction field
        V = VectorFunctionSpace(innermesh, "CG", 2)
        fname="output/spherical_traction_internal_lambda"+str(self.l1)\
            +"beta"+str(self.b1)+".pvd"
        vel=Function(V)
        w=spherical_traction_internal(a=self.a1,b=self.b1,l=self.l1,\
                            strain=self.E,element=V.ufl_element())
        vel.interpolate(w)
        v_out=File(fname)
        v_out<<vel
    def calculate_external_flow(self,outermesh):
        """Calculates velocity, pressure, compaction 
        and traction outside the sphere"""
        V = VectorFunctionSpace(outermesh, "CG", 2)
        U= FunctionSpace(outermesh, "CG", 2)
        # Internal flow
        # Write the compaction field for internal flow in pure shear
        fname="output/spherical_compaction_external_lambda"+str(self.l1)\
            +"beta"+str(self.b1)+".pvd"
        C=Function(U)
        u=spherical_compaction_external(a=self.a1,b=self.b1,l=self.l1\
                              ,strain=self.E,element=U.ufl_element())
        C.interpolate(u)
        C_out=File(fname)
        C_out <<C
        # Write the pressure field for external flow in pure shear
        fname="output/spherical_pressure_external_lambda"+str(self.l1)\
            +"beta"+str(self.b1)+".pvd"
        P=Function(U)
        u=spherical_pressure_external(a=self.a1,b=self.b1,l=self.l1,\
                            strain=self.E,element=U.ufl_element())
        P.interpolate(u)
        C_out=File(fname)
        C_out <<P
        # Write the normal gradient of pressure/melt flux
        # field for external flow in pure shear
        fname="output/spherical_flux_external_lambda"+str(self.l1)\
            +"beta"+str(self.b1)+".pvd"
        P=Function(U)
        u=spherical_flux_external(a=self.a1,b=self.b1,l=self.l1,\
                        strain=self.E,element=U.ufl_element())
        P.interpolate(u)
        C_out=File(fname)
        C_out <<P

        #Velocity field
        fname="output/spherical_velocity_external_lambda"+str(self.l1)\
            +"beta"+str(self.b1)+".pvd"
        vel=Function(V)
        w=spherical_velocity_external(a=self.a1,b=self.b1,l=self.l1,\
                            strain=self.E,element=V.ufl_element())
        vel.interpolate(w)
        v_out=File(fname)
        v_out<<vel

        #Traction field
        fname="output/spherical_traction_external_lambda"+str(self.l1)\
            +"beta"+str(self.b1)+".pvd"
        vel=Function(V)
        w=spherical_traction_external(a=self.a1,b=self.b1,l=self.l1,\
                            strain=self.E,element=V.ufl_element())
        vel.interpolate(w)
        v_out=File(fname)
        v_out<<vel

        

class coeffs:
    """This class computes and stores the 6 coefficients for the value of a=1"""
    def __init__(self,a,b,l):
        """Initiate the class"""
        self.a=a
        self.b=b
        self.l=l
        self.F=-0.156e2 * (self.l - 0.1e1) ** 2 \
            / (0.150e2 * self.b * self.l ** 3 + 0.126e2 * self.l ** 3\
               + 0.581e2 * self.b * self.l ** 2 + 0.120e3 * self.l ** 2 \
               + 0.728e2 * self.b * self.l + 0.170e3 * self.l\
               + 0.291e2 * self.b + 0.480e2)
        self.G=0.250e1 * (0.178e2 * self.b * self.l + 0.969e1 * self.b\
                          + 0.407e2 * self.l + 0.610e2 \
                          + 0.750e1 * self.b * self.l ** 2 \
                          - 0.317e2 * self.l ** 2) \
                          / (0.150e2 * self.b * self.l ** 3 \
                             + 0.126e2 * self.l ** 3 \
                             + 0.581e2 * self.b * self.l ** 2 \
                             + 0.120e3 * self.l ** 2 \
                             + 0.728e2 * self.b * self.l \
                             + 0.170e3 * self.l + 0.291e2 * self.b + 0.480e2)
        self.H= 0.147e4 * (self.l - 0.1e1) * (self.l + 0.842e0) \
                / (0.150e2 * self.b * self.l ** 3 + 0.126e2 * self.l ** 3\
                   + 0.581e2 * self.b * self.l ** 2 + 0.120e3 * self.l ** 2\
                   + 0.728e2 * self.b * self.l + 0.170e3 * self.l \
                   + 0.291e2 * self.b + 0.480e2)
        self.L=-0.125e2 * (self.b * self.l ** 2 + 0.238e1 * self.b * self.l\
                           + 0.838e0 * self.l ** 2 + 0.592e1 * self.l \
                           + 0.129e1 * self.b + 0.259e1) * (self.l - 0.1e1)\
                           / (0.150e2 * self.b * self.l ** 3 \
                              + 0.126e2 * self.l ** 3 + 0.581e2 * self.b \
                              * self.l ** 2 + 0.120e3 * self.l ** 2 \
                              + 0.728e2 * self.b * self.l + 0.170e3 * self.l\
                              + 0.291e2 * self.b + 0.480e2)
        self.M=-0.250e1 * (0.238e1 * self.b * self.l + 0.129e1 * self.b \
                           + 0.215e2 * self.l + 0.151e2 + self.b * self.l ** 2\
                           + 0.838e0 * self.l ** 2) * (self.l - 0.1e1)\
                           / (0.150e2 * self.b * self.l ** 3 \
                              + 0.126e2 * self.l ** 3 + 0.581e2 * self.b \
                              * self.l ** 2 + 0.120e3 * self.l ** 2 \
                              + 0.728e2 * self.b * self.l \
                              + 0.170e3 * self.l + 0.291e2 * self.b + 0.480e2)
        self.N=0.408e2 * (self.l - 0.1e1) * (self.l + 0.842e0) \
                / (0.150e2 * self.b * self.l ** 3 + 0.126e2 * self.l ** 3\
                   + 0.581e2 * self.b * self.l ** 2 + 0.120e3 * self.l ** 2\
                   + 0.728e2 * self.b * self.l + 0.170e3 * self.l\
                   + 0.291e2 * self.b + 0.480e2)



##########################################################
## The following classes contain specific functions for
## individual variables
########################################################
        
class spherical_velocity_internal(Expression):
    """Calculates internal velocity"""
    def __init__(self,a,b,l,strain,element):
        self.l=l
        self.a=a
        self.b=b
        self.E=strain
        self.element=element
    def eval(self,value,x):
        E=self.E
        coef=coeffs(self.a,self.b,self.l)
        rEr=x[0]*(E[0]*x[0]+E[3]*x[1]+E[4]*x[2])\
             +x[1]*(E[3]*x[0]+E[1]*x[1]+E[5]*x[2])\
             +x[2]*(E[4]*x[0]+E[5]*x[1]+E[2]*x[2])
        #Er=x
        value[0]=E[0]*x[0]+E[3]*x[1]+E[4]*x[2]
        value[1]=E[3]*x[0]+E[1]*x[1]+E[5]*x[2]
        value[2]=E[4]*x[0]+E[5]*x[1]+E[2]*x[2]
        Er=value
        r = np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
        term1=2.0*(coef.F*r**2+coef.G+coef.H*i2(r)/r**2)
        term2 = coef.H*i3(r)/r**3 - 4.0*coef.F/5
                
        value[0]=term1*Er[0]+term2*rEr*x[0]
        value[1]=term1*Er[1]+term2*rEr*x[1]
        value[2]=term1*Er[2]+term2*rEr*x[2]
        
class spherical_traction_internal(Expression):
    """Calculates traction inside the sphere"""
    def __init__(self,a,b,l,strain,element):
        self.l=l
        self.a=a
        self.b=b
        self.E=strain
        self.element=element
    def eval(self,value,x):
        E=self.E
        coef=coeffs(self.a,self.b,self.l)
        rEr=x[0]*(E[0]*x[0]+E[3]*x[1]+E[4]*x[2])\
             +x[1]*(E[3]*x[0]+E[1]*x[1]+E[5]*x[2])\
             +x[2]*(E[4]*x[0]+E[5]*x[1]+E[2]*x[2])
        #Er=x
        value[0]=E[0]*x[0]+E[3]*x[1]+E[4]*x[2]
        value[1]=E[3]*x[0]+E[1]*x[1]+E[5]*x[2]
        value[2]=E[4]*x[0]+E[5]*x[1]+E[2]*x[2]
        Er=value
        r = np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
        term1=2.0*self.l*(-19.0*coef.F/5.0 - 4.0*coef.H*i3(r)/r**3)
        term2 = 2.0*self.l*(16.0*coef.F*r**2/5+2.0*coef.G\
                            +2.0*coef.H*(i2(r)/r**2+i3(r)/r))
                

        value[0]=term1*rEr*x[0]+term2*Er[0]
        value[1]=term1*rEr*x[1]+term2*Er[1]
        value[2]=term1*rEr*x[2]+term2*Er[2]
        
class spherical_pressure_internal(Expression):
    """Calculates pressure internal to sphere of radius a"""
    def __init__(self,a,b,l,strain,element):
        self.l=l
        self.a=a
        self.b=b
        self.E=strain
        self.element=element
    def eval(self,value,x):
        E=self.E
        coef=coeffs(self.a,self.b,self.l)
        rEr=x[0]*(E[0]*x[0]+E[3]*x[1]+E[4]*x[2])\
             +x[1]*(E[3]*x[0]+E[1]*x[1]+E[5]*x[2])\
             +x[2]*(E[4]*x[0]+E[5]*x[1]+E[2]*x[2])
        r = np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
        term1=self.l*(42.0*coef.F/5.0+(self.b+2.0)*coef.H*i2(r)/r**2)
       
        value[0]=term1*rEr
        
class dP_boundary(Expression):
    """Calculates pressure internal to sphere of radius a"""
    def __init__(self,a,b,l,strain,element):
        self.l=l
        self.a=a
        self.b=b
        self.E=strain
        self.element=element
    def eval(self,value,x):
        E=self.E
        coef=coeffs(self.a,self.b,self.l)
        rEr=x[0]*(E[0]*x[0]+E[3]*x[1]+E[4]*x[2])\
             +x[1]*(E[3]*x[0]+E[1]*x[1]+E[5]*x[2])\
             +x[2]*(E[4]*x[0]+E[5]*x[1]+E[2]*x[2])
        r = np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
        if r > 1.0-DOLFIN_EPS or r < 1.0+DOLFIN_EPS:
           #internal pressure
            term1=self.l*(42.0*coef.F/5.0\
                          +(self.b+2.0)*coef.H*i2(r)/r**2)*rEr
            #external pressure
            term2=(6.0*coef.L/r**5+(self.b+2.0)*coef.N*k2(r)/r**2)*rEr
            value[0]=term1-term2 
        else:
            value[0]=0.0
        
class spherical_flux_internal(Expression):
    """Calculates the normal component of pressure gradient
    internal to sphere of radius a"""
    def __init__(self,a,b,l,strain,element):
        self.l=l
        self.a=a
        self.b=b
        self.E=strain
        self.element=element
    def eval(self,value,x):
        E=self.E
        coef=coeffs(self.a,self.b,self.l)
        rEr=x[0]*(E[0]*x[0]+E[3]*x[1]+E[4]*x[2])\
             +x[1]*(E[3]*x[0]+E[1]*x[1]+E[5]*x[2])\
             +x[2]*(E[4]*x[0]+E[5]*x[1]+E[2]*x[2])
        r = np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
        term1=self.l*(84.0*coef.F/5.0+(self.b+2.0)*coef.H*(2*i2(r)/r**2\
                                                           +i3(r)/r))
        value[0]=term1*rEr        
        
class spherical_compaction_internal(Expression):
    """Calculates compaction internal to sphere of radius a"""
    def __init__(self,a,b,l,strain,element):
        self.l=l
        self.a=a
        self.b=b
        self.E=strain
        self.element=element
    def eval(self,value,x):
        E=self.E
        coef=coeffs(self.a,self.b,self.l)
        rEr=x[0]*(E[0]*x[0]+E[3]*x[1]+E[4]*x[2])\
             +x[1]*(E[3]*x[0]+E[1]*x[1]+E[5]*x[2])\
             +x[2]*(E[4]*x[0]+E[5]*x[1]+E[2]*x[2])
        r = np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
        term1=coef.H*i2(r)/r**2
        value[0]=term1*rEr        
        
######################################################
########## Functions for flow in the substrate around the sphere
################################################
class spherical_velocity_external(Expression):
    """Calculates external velocity"""
    def __init__(self,a,b,l,strain,element):
        self.l=l
        self.a=a
        self.b=b
        self.E=strain
        self.element=element
    def eval(self,value,x):
        E=self.E
        coef=coeffs(self.a,self.b,self.l)
        rEr=x[0]*(E[0]*x[0]+E[3]*x[1]+E[4]*x[2])\
             +x[1]*(E[3]*x[0]+E[1]*x[1]+E[5]*x[2])\
             +x[2]*(E[4]*x[0]+E[5]*x[1]+E[2]*x[2])
        #Er=x
        value[0]=E[0]*x[0]+E[3]*x[1]+E[4]*x[2]
        value[1]=E[3]*x[0]+E[1]*x[1]+E[5]*x[2]
        value[2]=E[4]*x[0]+E[5]*x[1]+E[2]*x[2]
        Er=value
        r = np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
        term1=6.0*coef.M/r**5+2.0*coef.N*k2(r)/r**2+1
        term2 = 3.0*coef.L/r**5-15.0*coef.M/r**7-coef.N*k3(r)/r**3
                
        value[0]=term1*Er[0]+term2*rEr*x[0]
        value[1]=term1*Er[1]+term2*rEr*x[1]
        value[2]=term1*Er[2]+term2*rEr*x[2]
class spherical_traction_external(Expression):
    """Calculates traction outside the sphere"""
    def __init__(self,a,b,l,strain,element):
        self.l=l
        self.a=a
        self.b=b
        self.E=strain
        self.element=element
    def eval(self,value,x):
        E=self.E
        coef=coeffs(self.a,self.b,self.l)
        rEr=x[0]*(E[0]*x[0]+E[3]*x[1]+E[4]*x[2])\
             +x[1]*(E[3]*x[0]+E[1]*x[1]+E[5]*x[2])\
             +x[2]*(E[4]*x[0]+E[5]*x[1]+E[2]*x[2])
        #Er=x
        value[0]=E[0]*x[0]+E[3]*x[1]+E[4]*x[2]
        value[1]=E[3]*x[0]+E[1]*x[1]+E[5]*x[2]
        value[2]=E[4]*x[0]+E[5]*x[1]+E[2]*x[2]
        Er=value
        r = np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
        
        term1 = 2.0*(-12.0*coef.L/r**5+60.0*coef.M/r**7\
                     +4.0*coef.N*k3(r)/r**3)
        term2 = 2.0*(-24.0*coef.M/r**5+3.0*coef.L/r**3\
                     +2.0*coef.N*(k2(r)/r**2-k3(r)/r)+1.0)
        value[0]=term1*rEr*x[0]+term2*Er[0]
        value[1]=term1*rEr*x[1]+term2*Er[1]
        value[2]=term1*rEr*x[2]+term2*Er[2]
        
class spherical_pressure_external(Expression):
    """Calculates pressure external to sphere of radius a"""
    def __init__(self,a,b,l,strain,element):
        self.l=l
        self.a=a
        self.b=b
        self.E=strain
        self.element=element
    def eval(self,value,x):
        E=self.E
        coef=coeffs(self.a,self.b,self.l)
        rEr=x[0]*(E[0]*x[0]+E[3]*x[1]+E[4]*x[2])\
             +x[1]*(E[3]*x[0]+E[1]*x[1]+E[5]*x[2])\
             +x[2]*(E[4]*x[0]+E[5]*x[1]+E[2]*x[2])
        r = np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
        term1=6.0*coef.L/r**5+(self.b+2.0)*coef.N*k2(r)/r**2
        value[0]=term1*rEr
class spherical_flux_external(Expression):
    """Calculates normal component of pressure gradient
    external to sphere of radius a"""
    def __init__(self,a,b,l,strain,element):
        self.l=l
        self.a=a
        self.b=b
        self.E=strain
        self.element=element
    def eval(self,value,x):
        E=self.E
        coef=coeffs(self.a,self.b,self.l)
        rEr=x[0]*(E[0]*x[0]+E[3]*x[1]+E[4]*x[2])\
             +x[1]*(E[3]*x[0]+E[1]*x[1]+E[5]*x[2])\
             +x[2]*(E[4]*x[0]+E[5]*x[1]+E[2]*x[2])
        r = np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
        term1=-18.0*coef.L/r**5+(self.b+2.0)*coef.N*(2.0*k2(r)/r**2-k3(r)/r)
        value[0]=term1*rEr                
class spherical_compaction_external(Expression):
    """Calculates compaction external to sphere of radius a"""
    def __init__(self,a,b,l,strain,element):
        self.l=l
        self.a=a
        self.b=b
        self.E=strain
        self.element=element
    def eval(self,value,x):
        E=self.E
        coef=coeffs(self.a,self.b,self.l)
        rEr=x[0]*(E[0]*x[0]+E[3]*x[1]+E[4]*x[2])\
             +x[1]*(E[3]*x[0]+E[1]*x[1]+E[5]*x[2])\
             +x[2]*(E[4]*x[0]+E[5]*x[1]+E[2]*x[2])
        r = np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
        term1=coef.N*k2(r)/r**2
        value[0]=term1*rEr        



