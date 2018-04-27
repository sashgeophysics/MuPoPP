""" MuMaP Forwad
Copyright Saswata Hier-Majumder, 2017
Royal Holloway University of London

This module contains classes and functions relevant to calculate the
effect of melting on the elastic moduli and seismic wave velocities in 
partially molten rocks. To run the demonstrations, a working version of
Python 2.7.12 or higher, numpy, and matplotlib libraries are required.

Detailed description for each class is provided within the classes. 
This is an overview of the classes.

EOS: Contains functions for equations of state for solids and melts.
This class also contains the PREM model of Dziewonski and Anderson (1984). 
To initiate this class for a given material known values of some paramters 
need to be provided. In most cases, this class will be initiated from
either the Solid or the Melt class, which contain default values for the 
parameters.

Solid: This class contains a number of variables and functions relating
to the properties of the solid. All of the parameters are provided with
a default value. See the docstring for more details. By default, this class 
uses the PREM model to evaluate the elastic properties corresponding to the
depth parameter provided during instantiation.

Melt: This class contains physical paramters for calculating the EOS of the
melt. It also defaults to a dihedral angle of 15 degrees. The choice
of parameters for the EOS are set by the variable melt_comp. Please
see the docstring for __init__ for a full description of the currently
available choices.

Poroelasticity:  This class contains functions for calculating effective
elastic properties for a given melt fraction and physical properties
of the solid and the melt, contained in those classes. This class contains
three choices of contiguity models, see the docstrings for more information.
The default model is von Bargen and Waff, which allows for variation
in the idhedral angle. Do not use this model for higher melt fractions.

Example1: 
Plot Vinet equation of state of MORB using parameters
from Guillot and Sator(2007), run demo2.py for more plots
    >>> from mumap import *
    >>> import matplotlib.pylab as plt
    >>> rho1=np.linspace(2800.0,4000.0)
    >>> melt1=Melt(melt_comp=2,rho=rho1)
    >>> melt1.Melt_EOS.Vinet()
    >>> plt.plot(melt1.Melt_EOS.P/1.0e9,melt1.Melt_EOS.rho)
    >>> plt.show()

Example 2:
Plot the Vs/V0 as a function of melt fraction. In this case
all the melt resides in films of aspect ratio 0.01. This uses the formulation
of Walsh, 1969 
    >>> from mumap import*
    >>> import matplotlib.pylab as plt
    >>> phi=np.linspace(1.0e-3,0.15)
    >>> rock=Poroelasticity(phi=phi)
    >>> rock.Melt=Melt(rho=3200.0)
    >>> rock.film(aspect=0.01)
    >>> plt.plot(rock.meltfrac,rock.vs_over_v0)
    >>> plt.show()

Example 3:
Plot Vs/V0 as a function of melt fraction. In this case, the melt geometry
is calculated from von Bargen and Waff (1986) and the effective elastic
moduli are calculated following Hier-Majumder et al. (2014). Run demo4.py
for more information. 
    >>> from mumap import *
    >>> import matplotlib.pylab as plt
    >>> phi1=np.linspace(1.0e-3,0.15)
    >>> rock=Poroelasticity(phi=phi1)
    >>> rock.tube()
    >>> plt.show() 
"""


import numpy as np
import matplotlib.pylab as plt

def Moduli_to_Velocity(G,K,rho):
    """ This function calculates Vs and Vp from elastic modulie.
    Please use SI units
    Args:
         G :  Shear modulus in Pa
         K :  Bulk modulus in Pa
         rho: Density in kg/m^3
    Returns:
         vs : Shear wave speed in m/s
         vp : P wave speed in m/s
    """
    vs=np.sqrt(G/rho)
    vp=np.sqrt(K/rho+4.0*G/rho/3.0)
    return(vs,vp)
def Velocity_to_Moduli(vs,vp,rho):
    """ This utility subroutine calculates shear modulus G,
    bulk modulus K, and Poisson's ratio nu, from known
    shear and P wave velocities. Please use SI units.
    Args:
         vs : Shear wave speed in m/s
         vp : P wave speed in m/s
         rho: Density in kg/m^3
    Returns:
         G :  Shear modulus in Pa
         K :  Bulk modulus in Pa
         nu:  Poisson's ratio
         
    """
    G = rho*vs**2
    K = rho*(vp**2-(4.0*vs**2)/3.0)
    nu=(3.0*K-2.0*G)/(6.0*K+2.0*G)
    return(G,K,nu)
def PREM_Profile(depth):
        """ This function is used as the default EOS for solids,
        unless otherwise chosen. The values of vs and vp are calculated
        from the PREM model.
        See Table 1 of Dziewonski and Anderson, 1981 for reference
        Input:
              depth:  Desired depth ARRAY for calculation, in m
        Returns:
              rho  :  Density of the solid 
              vs   :  Shear wave speed of the solid (m/s)
              vp   :  P wave velocity of the solid  (m/s)
              G    :  Shear modulus of the solid  (Pa)
              K    :  Bulk modulus of the solid (Pa)
              nu   :  Poisson's ratio of the solid
        The returned variables are stored as class attributes
        """
        rho=np.zeros(depth.shape[0])
        vs=rho
        vp=rho
        for ii in range(0,depth.shape[0]):
            r=6371.0-depth[ii]/1.0e3
            x=r/6371.0      
            if r>3480.0 and r<=3630.0:
                rho[ii]=7.9565-6.4761*x+5.5283*x**2-3.0807*x**3
                vp[ii]=15.3891-5.3181*x+5.5242*x**2-2.5514*x**3
                vs[ii]=6.9254+1.4672*x-2.0834*x**2+0.9783*x**3           
            elif r>3630.0 and r<=5600.0:
                rho[ii]=7.9565-6.4761*x+5.5283*x**2-3.0807*x**3
                vp[ii]=24.952-40.4673*x+51.4832*x**2-26.6419*x**3
                vs[ii]=11.671-13.7818*x+17.4575*x**2-9.2777*x**3
            elif r>5600.0 and r<=5701.0:
                rho[ii]=7.9565-6.4761*x+5.5283*x**2-3.0807*x**3
                vp[ii]=29.2766-23.6027*x+5.5242*x**2-2.5514*x**3
                vs[ii]=22.3459-17.2473*x-2.0834*x**2+0.9783*x**3
            elif r>5701.0 and r<=5771.0:
                rho[ii]=5.3197-1.4836*x
                vp[ii]=19.0957-9.8672*x
                vs[ii]=9.9839-4.9324*x
            elif r>5771.0 and r<=5971.0:
                rho[ii]=11.2494-8.0298*x
                vp[ii]=39.7027-32.6166*x
                vs[ii]=22.3512-18.5856*x
            elif r>5971.0 and r<=6151.0:
                rho[ii]=11.2494-8.0298*x
                vp[ii]=39.7027-32.6166*x
                vs[ii]=22.3512-18.5856*x
            elif r>6151.0 and r<=6346.6:
                rho[ii]=2.691+0.6924*x
                vp[ii]=4.1875+3.9382*x #effective isotropic velocity
                vs[ii]=2.1519+2.3481*x #effective isotropic velocity
            elif r>6346.6 and r<=6356.0:
                rho[ii]=2.9
                vp[ii]=6.8
                vs[ii]=3.9
            elif r>6356.0 and r<=6368.0:
                rho[ii]=2.6
                vp[ii]=5.8
                vs[ii]=3.2
            elif r>6368.0 and r<=6371.0:
                rho[ii]=1.02
                vp[ii]=1.45
                vs[ii]=0.0  
            else :
                rho[ii]=0.0
                vp[ii]=0.0
                vs[ii]=0.0
                print 'PREM:Depth out of solid range'
        
        rho=rho*1.0e3
        vs=vs*1.0e3
        vp=vp*1.0e3
        [G,K,nu]= Velocity_to_Moduli(vs,vp,rho)
        return(vs,vp,rho,G,K,nu)
             
class EOS:
    """This class contains equations of state
    to calculate density and elastic
    moduli of solids and melts
    
    To initiate this class, the following parameters are needed
    Parameters:
         K0   : Bulk modulus of the material at surface, in Pa 
         rho  : Density (kg/m^3) at the desired condition can be an array
         Kp   : Pressure derivative of the bulk modulus
         rho0 : Density (kg/m^3) at surface
    These parameters are used by Vinet and BM3 (Third order Birch-Murnaghan
    EOS), can be used for both solid and melt. The EOS PREM is for solids
    only. This class is instantiated within classes Melt and Solid
    """
    def __init__(self,K0,rho,Kp,rho0):
        """Initiate the class with known values of density, rho, surface
        bulk modulus, K0, and the pressure derivative
        of the bulk modulus, Kp."""
        self.K0=K0
        self.rho=rho
        self.Kp=Kp
        self.rho0=rho0
        self.P=0.0
    def Vinet(self):
        """This function uses the Vinet EOS to calculate
        the bulk modulus and pressure for a given density.
        Inputs:
             rho  : Density 
             rho0 : Surface density
             Kp   : Pressure derivative of bulk modulus
             K0   : Bulk modulus at surface
        These inputs are created during instantiation of the class
        Returns:
             P    : Pressure in Pa
             K    : Bulk modulus in Pa
        Both of these are stored as public attributes 
        """       
        z=self.rho/self.rho0
        self.P=3.0*self.K0*z**(2.0/3.0)*(1.0-z**(-1.0/3.0))\
           *np.exp(1.5*(self.Kp-1.0)*(1.0-z**(-1.0/3.0)))
        term1=3.0 *self.K0*np.exp(1.5*(self.Kp-1.0)*(1.0-z**(-1.0/3.0)))
        term2=z**(-1.0/3.0)*(2.0-z**(-1.0/3.0))/3.0 
        term3=0.5 *z**(-2.0/3.0)*(1.0-z**(-1.0 /3.0 ))*(self.Kp-1.0)
        dpdz=term1*(term2+term3)
        self.K=z*dpdz
    def BM3(self):
        """This function uses the third order Birch-Murnaghan EOS
        to calculate the bulk modulus and pressure for a given density.        
        Inputs:
             rho  : Density 
             rho0 : Surface density
             Kp   : Pressure derivative of bulk modulus
             K0   : Bulk modulus at surface
        These inputs are created during instantiation of the class
        Returns:
             P    : Pressure in Pa
             K    : Bulk modulus in Pa
        Both of these are stored as public attributes
        """
        z=self.rho/self.rho0
        self.P=1.5*self.K0*((self.rho/self.rho0)**(7.0/3.0)\
                        -(self.rho/self.rho0)**(5.0/3.0))\
                        *(1.0+0.75*(self.Kp-4.0)*((self.rho/self.rho0)\
                        **(2.0/3.0)-1.0))
        term1=0.75*self.K0*(self.Kp-4.0)*(z**2-z**(4.0/3.0))
        term2=1.0+0.75*(self.Kp-4.0)*(z**(2.0/3.0)-1.0)
        term3=0.5*self.K0*(7.0*z**(4.0/3.0)-5.0*z**(2.0/3.0))
        dpdz=term1+term2*term3
        self.K=z*dpdz
    def PREM(self,depth=60.0e3):
        """ This function is used as the default EOS for solids,
        unless otherwise chosen. The values of vs and vp are calculated
        from the PREM model.
        See Table 1 of Dziewonski and Anderson, 1981 for reference
        Input:
              depth:  Desired depth for calculation, in m
        Returns:
              rho  :  Density of the solid 
              vs   :  Shear wave speed of the solid (m/s)
              vp   :  P wave velocity of the solid  (m/s)
              G    :  Shear modulus of the solid  (Pa)
              K    :  Bulk modulus of the solid (Pa)
              nu   :  Poisson's ratio of the solid
        The returned variables are stored as class attributes
        """
        r=6371.0-depth/1.0e3
        x=r/6371.0
        if r>3480.0 and r<=3630.0:
            rho=7.9565-6.4761*x+5.5283*x**2-3.0807*x**3
            vp=15.3891-5.3181*x+5.5242*x**2-2.5514*x**3
            vs=6.9254+1.4672*x-2.0834*x**2+0.9783*x**3           
        elif r>3630.0 and r<=5600.0:
            rho=7.9565-6.4761*x+5.5283*x**2-3.0807*x**3
            vp=24.952-40.4673*x+51.4832*x**2-26.6419*x**3
            vs=11.671-13.7818*x+17.4575*x**2-9.2777*x**3
        elif r>5600.0 and r<=5701.0:
            rho=7.9565-6.4761*x+5.5283*x**2-3.0807*x**3
            vp=29.2766-23.6027*x+5.5242*x**2-2.5514*x**3
            vs=22.3459-17.2473*x-2.0834*x**2+0.9783*x**3
        elif r>5701.0 and r<=5771.0:
            rho=5.3197-1.4836*x
            vp=19.0957-9.8672*x
            vs=9.9839-4.9324*x
        elif r>5771.0 and r<=5971.0:
            rho=11.2494-8.0298*x
            vp=39.7027-32.6166*x
            vs=22.3512-18.5856*x
        elif r>5971.0 and r<=6151.0:
            rho=11.2494-8.0298*x
            vp=39.7027-32.6166*x
            vs=22.3512-18.5856*x
        elif r>6151.0 and r<=6346.6:
            rho=2.691+0.6924*x
            vp=4.1875+3.9382*x #effective isotropic velocity
            vs=2.1519+2.3481*x #effective isotropic velocity
        elif r>6346.6 and r<=6356.0:
            rho=2.9
            vp=6.8
            vs=3.9
        elif r>6356.0 and r<=6368.0:
            rho=2.6
            vp=5.8
            vs=3.2
        elif r>6368.0 and r<=6371.0:
            rho=1.02
            vp=1.45
            vs=0.0  
        else :
            rho=0.0
            vp=0.0
            vs=0.0
            print 'PREM:Depth out of solid range'
       
            
        self.rho=rho*1.0e3
        self.vs=vs*1.0e3
        self.vp=vp*1.0e3
        [self.G,self.K,self.nu]= Velocity_to_Moduli(self.vs,self.vp,self.rho)  
             
        
class Solid:
    """This class contains functions and properties of the solid.
    To initiate this class, the following parameters are needed
    Parameters:
        K0    : Bulk modulus of the material at surface, in Pa 
        Kp    : Pressure derivative of the bulk modulus
        G0    : Shear modulus of the material at surface, in Pa 
        Gp    : Pressure derivative of the shear modulus
        rho0  : Density (kg/m^3) at surface
        depth : m, below the surface
    By default, solid parameters are evaluated using the PREM
    model.
    """
    def __init__(self,K0=126.3e9,Kp=4.28, G0=78.0e9,Gp=1.71,\
                 rho0=2600.0,depth=60.0e3):
        """initiates the class, assigns the values of K0,Kp,G0, and Gp
        from Abramson et al (1997), Table 4.1 of Poirier's Introduction
        to the physics of the Earth's interior 2nd ed.
        rho0 is from Prem model. Depth is given 60 km by default
        On initiation, K,G, and nu are given surface values"""
        self.K0=K0
        self.K=K0
        self.Kp=Kp
        self.G0=G0
        self.G=G0
        self.rho0=rho0
        self.rho=rho0
        self.depth=depth
        
        #Initiate the solid EOS
        self.Solid_EOS=EOS(self.K0,self.rho,self.Kp,self.rho0)
        #Now calculate K,G,nu, and rho from the PREM model
        self.Solid_EOS.PREM(depth=self.depth)
        self.K=self.Solid_EOS.K
        self.G=self.Solid_EOS.G
        self.nu=self.Solid_EOS.nu
        self.rho=self.Solid_EOS.rho
class Melt:
    """This class contains functions and propeties
    of the melt. To initiate the class, the following paramters
    are needed
    Parameters:
        theta     : degrees, solid-melt dihedral angle
        rho       : Melt density (kg/m^3)
        melt_comp : Melt composition for the EOS
    Pressure of the melt is calculated by using the EOS
    of choice using the melt_comp variable.
    
    """
    def __init__(self,theta=20.0,rho=3000.0,melt_comp=1):
        """ This function sets the properties of the melt
         See Table 1 of Wimert and Hier-Majumder(2012) for details
        The choices are 
            1 = peridotite melt, (2273 K) Guillot and Sator (2007)
            2 = MORB,(2073 K) Guillot and Sator  (2007)
            3 = peridotite melt, Ohtani and Maeda (2001)
            4 = MORB, Ohtani and Maeda (2001)
            5= peridotite+5%CO2 from Ghosh etal (2007)
        Any other value defaults to 1
        The optional value of dihedral angle
        is the second input. """
        self.theta=theta
        if melt_comp==2:
            self.K0=15.5e9 
            self.Kp=7.2 
            self.rho0=2590.0
            self.description='MORB,(2073 K) Guillot and Sator  (2007)'
        elif melt_comp==3:
            self.K0=24.9e9 
            self.Kp=6.4 
            self.rho0=2610.0
            self.description='peridotite melt, Ohtani and Maeda (2001)'
        elif melt_comp==4:
            self.K0=18.1e9 
            self.Kp=5.5 
            self.rho0=2590.0
            self.description='MORB, Ohtani and Maeda (2001)'
        elif melt_comp==5:
            self.K0=24.9e9 
            self.Kp=5.1 
            self.rho0=2670.0
            self.description='peridotite+5%CO2 from Ghosh etal (2007)'
        else:
            self.K0=16.5e9 
            self.Kp=7.2 
            self.rho0=2610.0 
            self.description='peridotite melt, (2273 K)\
            Guillot and Sator (2007)'
        
        self.rho=rho
        self.Melt_EOS=EOS(self.K0,self.rho,self.Kp,self.rho0)
        self.P=self.Melt_EOS.P
       

class Poroelasticity():
    """This class contains functions and variables
    relevant to calculating the effective properties
    for different melt geometries. There are 3
    different functions for calculating contiguity,
    the ratio between the area of grain-grain contact and
    the surface area of a grain. The result is saved
    in a variable, self.Contiguity.

    On initiation, the following variables are created
    Parameters
        theta      : dihedral angle
        melfrac    : Melt volume fraction (between 0 and 1)
        Contiguity : Area fraction of grain-grain contact
        Melt       : Object melt is created with theta
        Solid      : Object solid is created
    """
    def __init__(self,theta=20.0,phi=0.15):
        """Initiates the melt geometry class
        theta is the dihedral angle"""
        self.theta = theta
        self.meltfrac=phi
        self.Contiguity=0.0
        self.Melt=Melt(theta=self.theta)
        self.Solid=Solid()
                   
    def WHM12(self):
        """ This function returns the contiguity as a function of
        melt fraction. The dihedral angle, even taken as an input
        is currently not used, as the model of Wimert and
        Hier-Majumder is valid for a constant dihedral angle of
        approximately 30 degreees. Doesn't work well beyond melt
        volume fraction of 0.25.
        Parameters:
            meltfrac    : melt volume fraction
        Returns:
            Contiguity  : Fractional area of grain-grain contact
        """
        p1=-8065.0
        p2=6149.0
        p3=-1778.0
        p4=249.0
        p5=-19.77
        p6=1.0
        self.Contiguity=p1*self.meltfrac**5.0+p2*self.meltfrac**4.0\
                    +p3*self.meltfrac**3.0+p4*self.meltfrac**2.0\
                    +p5*self.meltfrac+p6
    def HMRB06(self):
        """ This function returns the two dimensional
        measurement of contiguity from Hier-Majumder
        et al (2006). This is not recommended as it typically
        returns contiguity values higher than the 3D models. 
        Parameters:
            meltfrac    : melt volume fraction
            theta       : semidihedral angle
        Returns:
            Contiguity  : Fractional area of grain-grain contact
        """
        temp2=self.theta*np.pi/90.0  
        temp1=np.sqrt(np.cos(temp2)**2-np.sin(temp2)*np.cos(temp2)\
                      -0.25*np.pi+temp2)
        chi1=np.abs(0.5*np.pi -2.0*temp2)/temp1
        chi2=np.abs(np.cos(temp2)-np.sin(temp2))/temp1
        #Area of grain-melt contact
        Agm=chi1*np.sqrt(self.meltfrac)
        #Area of grain-grain contact
        Agg= 1.0-chi2*np.sqrt(self.meltfrac)
        self.Contiguity=Agg/(Agg+Agm) 

    def VBW86(self):
        """ This function returns the contiguity
        of a partially molten unit cell as a function
        of melt volume fraction and dihedral angle,
        using the parametrization of von Bargen and
        Waff (1986). Notice that the parameter Agg
        shouldn't become zero at zero melt fraction,
        as erroneously indicated in their article.
        it is fixed by subtracting it from pi to match their
        Figure 10. Doesn't work
        beyond melt volume fraction fo 0.18. 
        Parameters:
            meltfrac    : melt volume fraction
            theta       : semidihedral angle
        Returns:
            Contiguity  : Fractional area of grain-grain contact
        """
        
        bss=np.array([8.16, -7.71e-2, 1.03e-3])
        pss=np.array([0.424, 9.95e-4, 8.6645e-6])
        bsl=np.array([12.86, -7.85e-2, 1.0043e-3])
        psl =np.array([0.43, 8.63e-5, 2.41e-5])

        b=bss[2]*self.theta**2+bss[1]*self.theta+bss[0]
        p=pss[2]*self.theta**2+pss[1]*self.theta+pss[0]
        #Area of grain-grain contact, adjusted (see above)
        Agg=np.pi-b*self.meltfrac**p 
        
        b=bsl[2]*self.theta**2+bsl[1]*self.theta+bsl[0] 
        p=psl[2]*self.theta**2+psl[1]*self.theta+psl[0]
        #Area of grain-melt contact
        Agm=b*self.meltfrac**p       
        self.Contiguity=2.0*Agg/(2.0*Agg+Agm)
        
    def set_contiguity(self,contiguity_model=1):
        """Calculates contiguity, default is VBW86"""
        if contiguity_model==1:
            self.VBW86()            
        elif contiguity_model==2:
            self.WHM12()
        else:
            self.HMRB06()
        

    def film(self,aspect=0.01):
        """ This function calculates Vs/Vs0,Vp/VP0, K/K0, and G/G0
        Based on the model of Walsh, 1969. On input, the aspect
        ratio is the ratio between the minor and major axis 
        of the elliptical melt inclusion.
        Subscript 1 is the stronger phase.
        On return, the four members in the array
        elastic_melt are respectively
        Vs/Vs0,Vp/VP0, K/K0, and G/G0. Please only enter SI units. 
        Input:
            aspect         : Aspect ratio of melt film
        Parameters
            self.Solid.K   : Solid bulk modulus
            self.Solid.G   : Solid shear modulus
            self.Solid.rho : Density of the solid
            self.meltfrac  : Melt volume fraction
        Returns:
            vs_over_v0     : Normalized shear velocity
            vp_over_v0     : Normalized P wave velocity
            K_over_K0      : Normalized bulk modulus
            G_over_G0      : Normalized shear modulus
            
        """
        
        K1=self.Solid.K
        mu1=self.Solid.G
        rho1=self.Solid.rho
        self.Melt.Melt_EOS.Vinet()
        K2=np.max(self.Melt.Melt_EOS.K)
        mu2=0.0
        rho2=np.max(self.Melt.Melt_EOS.rho)
        meltfrac=self.meltfrac

        [vs0,vp0]=Moduli_to_Velocity(mu1,K1,rho1)
        term3=3.0*aspect*np.pi*mu1*(3.0*K1+mu1)/(3.0*K1+4.0*mu1)
        term4=3.0*aspect*np.pi*mu1*(3.0*K1+2.0*mu1)/(3.0*K1+4.0*mu1)
        term1=meltfrac*(K1-K2)*(3.0*K1+4.0*mu2)/(3.0*K2+4.0*mu2+term3)/K1
        term2=meltfrac*(1+8.0*mu1/(4.0*mu2+term4)\
                        +2.0*(3.0*K2+2.0*mu2+2.0*mu1)\
                        /(3.0*K2+4.0*mu2+term3))*(mu1-mu2)/mu1/5.0
        Keff=1.0/(1.0+term1)
        Geff=1.0/(1.0+term2)        
        rho_av=(1.0-meltfrac)*rho1+meltfrac*rho2
        [vs,vp]=Moduli_to_Velocity(Geff*mu1,Keff*K1,rho_av)
        self.vs_over_v0=vs/vs0
        self.vp_over_v0=vp/vp0
        self.K_over_K0=Keff
        self.G_over_G0=Geff
        self.aspect=aspect

    def tube(self):
        """  
        This function calculates the S and P wave velocity reduction
        following the model of Hier-Majumder et al. (2014)
        Inputs:
              meltfrac            : Melt fraction in the rock
              contiguity          : contiguity of the rock
              Solid.K             : Bulk modulus of solid
              Solid.G             : Shear modulus of solid
              Solid.rho           : Density of the solid
              Solid.nu            : Poisson's ratio of the Solid
              Melt.Melt_EOS.K     : Bulk modulus of the melt
              Melt.Melt_EOS.rho   : Density of the melt
              
              
        Returns: 
              vs_over_v0          : Dimensionless normalized Vs
              vp_over_v0          : Dimensionless normalized vp
              K_over_K0           : Dimensionless normalized bulk modulus
              G_over_G0           : Dimensionless normalized shear modulus
        All variables are stored as public attributes to the class.
       
        """
        K=self.Solid.K
        G=self.Solid.G
        rho=self.Solid.rho
        self.Melt.Melt_EOS.Vinet()
        Kl=np.max(self.Melt.Melt_EOS.K)
        
        
        mu2=0.0
        rhol=np.max(self.Melt.Melt_EOS.rho)
        meltfrac=self.meltfrac
        self.set_contiguity()
        nu=self.Solid.nu
        
        a1=1.8625+0.52594*nu-4.8397*nu**2
        a2=4.5001-6.1551*nu-4.3634*nu**2
        a3=-5.6512+6.9159*nu+29.595*nu**2-58.96*nu**3
        b1=1.6122+0.13527*nu
        b2=4.5869+3.6086*nu
        b3=-7.5395-4.8676*nu-4.3182*nu**2
        #Equation A5
        m=a1*self.Contiguity+a2*(1.0-self.Contiguity)\
           +a3*((1.0-self.Contiguity)**1.5)*self.Contiguity 
        #Equation A6
        n=b1*self.Contiguity+b2*(1.0-self.Contiguity)\
           +b3*self.Contiguity*(1.0-self.Contiguity)**2 

        h=1.0-(1.0-self.Contiguity)**m    #Eq A3
        gg=1.0-(1.0-self.Contiguity)**n   # Eq A4

        #normalized bulk modulus of the skeletal framework eq 27, H-m 2008
        KboverK=(1.0-meltfrac)*h        #Eq A1 divided by k
        G1=(1.0-meltfrac)*gg            #Eq A2 divided by mu

        bet=K/Kl
        gam=G/K
        rr0=rhol/rho

        K1=KboverK+((1.0-KboverK)**2)/(1.0-meltfrac-KboverK+meltfrac*bet)
        rhobaroverrho=1.0-meltfrac+rr0*meltfrac

        vs1=np.sqrt(G1)/np.sqrt(rhobaroverrho)

        vp1=np.sqrt(K1+4.0*gam*G1/3.0)/np.sqrt(1.0+4.0*gam/3.0)/np.sqrt(rhobaroverrho)
        self.vs_over_v0=vs1
        self.vp_over_v0=vp1
        self.K_over_K0=K1
        self.G_over_G0=G1

    def HH2000_Film(self):
       """This function calculates the S and P wave velocity reduction
       following the model of Paradigm 1 of Hammonds and Humphreys (2000)
       It assumes relaxed, random cuspate melt geometry. See Table 2
       of Hammond and Humphreys (2000) for details.
       Inputs:
              meltfrac   : Melt fraction in the rock
       Returns: 
              vs_over_v0 : Dimnesionless normalized Vs
              vp_over_v0 : Dimensionless normalized vp
             
       All variables are stored as public attributes to the class
       """
       self.vs_over_v0 = -0.064*self.meltfrac
       self.vp_over_v0 = -0.029*self.meltfrac
    def HH2000_combined(self):
        """This function calculates the S and P wave velocity reduction
        following the model of Paradigm 2 of Hammonds and Humphreys (2000)
        It assumes relaxed, random cuspate melt geometry. See Table 2
        of Hammond and Humphreys (2000) for details.
        Inputs:
               meltfrac   : Melt fraction in the rock
        Returns: 
               vs_over_v0 : Dimnesionless normalized Vs
               vp_over_v0 : Dimensionless normalized vp
        
        All variables are stored as public attributes to the class"""

        self.vs_over_v0 = -0.11*self.meltfrac
        self.vp_over_v0 = -0.063*self.meltfrac

        
