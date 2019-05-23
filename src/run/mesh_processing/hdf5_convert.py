# Run this script on 1 core for the target .xml mesh file to generate a .h5 version
# Then use the generated .h5 file for simulation in parallel

from dolfin import *

mesh_fname = "berea_subvol_FECFDCAD_ASCII_surface.xml"
mesh = Mesh(mesh_fname);

boundaries_fname = "berea_subvol_FECFDCAD_ASCII_surface_facet_region.xml"
boundaries = MeshFunction('size_t', mesh, boundaries_fname);

extension = ".h5"
hdf = HDF5File(mesh.mpi_comm(), "hdf5_" + mesh_fname + extension, "w")
hdf.write(mesh, "/mesh")
hdf.write(boundaries, "/boundaries")
