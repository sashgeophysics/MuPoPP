###### Generates a mesh from a .xml file ######

# Either #

#Export from PerGeos for abaqus as .inp file
#Run dolfin-convert filename.inp filename.xml
#Run this code

# Or #

#PerGeos tutorial > advanced surface and grid generation
#After the first remeshing step fix the intersections and make sure aspect ratios are ok
#Export as .stl
#tetgen -Vpgq1.2/0 fname.stl to get .mesh
#dolfin-convert .mesh to get .xml
#Run this code

#####################################################################################
from dolfin import*
import matplotlib.pylab as plt

fname="berea_subvol_FECFDCAD_ASCII_surface.xml"
mesh=Mesh(fname)

output_dir = "mesh_output/"
extension = ".pvd"
mesh_out = File(output_dir + fname + extension, "compressed")

mesh_out << mesh


#plt.figure()
#plot(mesh)
#plt.show()
