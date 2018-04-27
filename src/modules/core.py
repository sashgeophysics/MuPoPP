#!/usr/bin/env python

# ======================================================================
# core.py
#
# Contains routines for data input and output, and some other basic
# functions.
#
# Original version from Authors:
# Laura Alisic, University of Cambridge
# Sander Rhebergen, University of Oxford
# Garth Wells, University of Cambridge
# John Rudge, University of Cambridge
#
# Last modified: 6 Jun 2015 by Andy Cai
# ======================================================================

from dolfin import *
import math, sys, os, string
 
# ======================================================================

def u_max(U, cylinder_mesh):
    """Return |u|_max for a U = (u, p) systems"""

    mesh   = U.function_space().mesh()
    V      = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    v      = TestFunction(V)

    if cylinder_mesh:
        u, p, o = U.split()
    else:
        u, p    = U.split()

    volume = v.cell().volume
    L      = v*sqrt(dot(u, u))/volume*dx
    b      = assemble(L)

    return b.norm("linf")

# ======================================================================

def compute_dt(U, cfl, h_min, cylinder_mesh):
    """Compute time step dt"""

    umax   = u_max(U, cylinder_mesh)

    return cfl*h_min/max(1.0e-6, umax)

# ======================================================================

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

# ======================================================================

# Read in parameter file, store entries in dictionary
def parse_param_file(filename):
    """Parse a 'key = value' style control file, and return a dictionary of those settings"""

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

# ======================================================================

def write_vtk(Q, p, phi, u, v0, shear_visc, bulk_visc, perm, vel_pert_out, velocity_out, \
              pressure_out, porosity_out, divU_out, shear_visc_out, bulk_visc_out, perm_out):
    """Write vector and scalar fields to files"""
    # Note: Renaming fields so that they are easier to distinguish in Paraview

    # Velocity field
    u.rename("velocity", "")
    velocity_out  << u

    # Velocity perturbation    
    vel_pert      = u - v0
    vel_pert_proj = project(vel_pert)
    vel_pert_proj.rename("vel_perturbation", "")
    vel_pert_out  << vel_pert_proj

    # Divergence of velocity == compaction
    divU_proj     = project(div(u))
    divU_proj.rename("div_u", "")
    divU_out      << divU_proj

    # Pressure field
    p.rename("pressure", "")
    pressure_out  << p
    
    # Porosity field
    phi.rename("porosity", "")
    porosity_out  << phi

    # Rheology fields
    shear_visc_proj = project(shear_visc, Q)
    shear_visc_proj.rename("shear_viscosity", "")
    shear_visc_out  << shear_visc_proj
    bulk_visc_proj  = project(bulk_visc, Q)
    bulk_visc_proj.rename("bulk_viscosity", "")
    bulk_visc_out   << bulk_visc_proj
    perm_proj       = project(perm, Q)
    perm_proj.rename("permeability", "")
    perm_out        << perm_proj

# ======================================================================

# EOF
