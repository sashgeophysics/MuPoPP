# MuPoPP (Multiphase Porous flow and Physical Properties) v 1.2.1

This module contains functions for solving mass and moemntum conservation
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

#Classes

This module contains a number of classes corresponding to different PDE problems. Some of the classes may not work properly. See below for a description of the classes that are currently working.

# Class compaction

   This class calculates the governing equations for a
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

# Class darcy_two_phase (To be updated)
    This class contains the variables and functions
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
    and p is the modified pressure

# Class  DarcyAdvection

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

# Class StokesAdvection (To be updated)

  	  
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

# Class CCS

    This class solves for a simple advection-diffusion
    equation for a single or multicomponent flow, the governing
    PDEs for Darcy flow are:        
    div(u) = 0                                                          (1)
    phi*u = -k*(grad(p)-drho*zhat)                                      (2)
    dc0/dt + dot(u,grad(c0)) = div(grad(c0))/Pe - Da*c0*c1/phi + beta*f (3)
    dc1/dt = -fac1*Da*c0*c1/phi                                             (4)
    and 
    dc2/dt = fac2*Da*c0*c1/phi
    where c0 and c1 are concentrations of the reactants in the liquid
    and solid, u is the fluid velocity, p is pressure, k is permeability
    drho=difference between liquid and solid densities, zhat is a unit
    vecotr in vertically upward direction, Pe is Peclet number, Da is
    the Dahmkoler number, beta is source strength, f is a function
    for lateral variations in source of c0, and phi is the constant porosity.
    c2 is the concentration of solid product, fac1 and fac2 are factors
    to convert from volume fraction to mass fraction. The underlying chemical
    reaction is
    An + H2CO3 = Ka + CaCO3
    c0 is the consentration of H2CO3 in the liquid
    c1 is the concentration of An in the solid
    c2 is the concentration of CaCO3 in the solid

    On initiation of the class:
              the dimensionless numbers, Pe, Da, alpha =beta*phi are loaded
              the timestep dt and CFL criterion are also loaded to default
              values.
    For the remaining functions, see the docstring of each individual function
    for help.

