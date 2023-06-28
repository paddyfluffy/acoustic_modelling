import sys
import os
import numpy as np

try:
    import gmsh
except ImportError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)


from dolfinx.io import XDMFFile, gmshio

from mpi4py import MPI


## create sphere mesh
## sphere mesh

gmsh.initialize()

# Choose if Gmsh output is verbose
gmsh.option.setNumber("General.Terminal", 0)
model = gmsh.model()
model.add("Sphere")
model.setCurrent("Sphere")
model.mesh.setSize(gmsh.model.getEntities(0), 0.0017)
# mesh_01 = model.occ.addSphere(0, 0, 0, 1, tag=1)
mesh_01 = model.occ.addBox(0, 0, 0, 2, 1, 1)

# Synchronize OpenCascade representation with gmsh model
model.occ.synchronize()

# Add physical marker for cells. It is important to call this function
# after OpenCascade synchronization
model.add_physical_group(dim=3, tags=[mesh_01])


# Generate the mesh
model.mesh.generate(dim=3)

# Create a DOLFINx mesh (same mesh on each rank)
msh, cell_markers, facet_markers = gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
msh.name = "Sphere"
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

gmsh.finalize()

# msh, cell_tags, facet_markers = gmshio.read_from_msh("mesh3D.msh", MPI.COMM_WORLD, 0, gdim=3)


vertices = msh.geometry.x
# Compute the minimum and maximum values
min_x = np.min(vertices[:, 0])
max_x = np.max(vertices[:, 0])

min_y = np.min(vertices[:, 1])
max_y = np.max(vertices[:, 1])

min_z = np.min(vertices[:, 2])
max_z = np.max(vertices[:, 2])

# Print the results
print("Minimum and maximum values:")
print("X: ", min_x, " ; ", max_x)
print("Y: ", min_y, " ; ", max_y)
print("Z: ", min_z, " ; ", max_z)

## dolfinx

import numpy as np
import ufl
from dolfinx import geometry
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form, Constant
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square
from ufl import dx, grad, inner, Measure
from mpi4py import MPI
from petsc4py import PETSc

# import dolfinx.log

# # Set the logging level to capture detailed information
# dolfinx.log.set_log_level(dolfinx.log.LogLevel.DEBUG)

# # Enable logging to the console
# dolfinx.log.set_console_output(True)


Z_s = Constant(msh, PETSc.ScalarType(1))

ds = Measure("ds", domain=msh, subdomain_data=facet_markers)

# approximation space polynomial degree
deg = 2

#frequency range definition
f_axis = np.arange(50, 2005, 5)

#Mic position definition
mic = np.array([0.05, 0.05, 0.05])

# Test and trial function space
V = FunctionSpace(msh, ("CG", deg))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = Function(V)

# Source amplitude
Q = 0.0001

#Source definition position = (Sx,Sy, Sz)
Sx = 0.2
Sy = 0.2
Sz = 0.2

#Narrow normalized gauss distribution
alfa = 0.015
delta_tmp = Function(V)
# delta_tmp.interpolate(lambda x : 1/(np.abs(alfa)*np.sqrt(np.pi))*np.exp(-(((x[0]-Sx)**2+(x[1]-Sy)**2)/(alfa**2))))
delta_tmp.interpolate(lambda x: 1 / (np.abs(alfa) * np.sqrt(np.pi)) * np.exp(-(((x[0] - Sx) ** 2 + (x[1] - Sy) ** 2 + (x[2] - Sz) ** 2) / (alfa ** 2))))
# delta_tmp.interpolate(lambda x: 1 / (np.abs(alfa) * (2 * np.pi) ** (3/2)) * np.exp(-(((x[0] - Sx) ** 2 + (x[1] - Sy) ** 2 + (x[2] - Sz) ** 2) / (2 * alfa ** 2))))
int_delta_tmp = assemble_scalar(form(delta_tmp*dx)) #form() l'ho scoperto per sbaglio. Senza esplode.
delta = delta_tmp/int_delta_tmp


# Source amplitude
Q = 0.0001

#Source definition position = (Sx,Sy)
Sx = 0.2
Sy = 0.2
Sz = 0.2

#Narrow normalized gauss distribution
alfa = 0.015
delta_tmp = Function(V)
delta_tmp.interpolate(lambda x : 1/(np.abs(alfa)*np.sqrt(np.pi))*np.exp(-(((x[0]-Sx)**2+(x[1]-Sy)**2)/(alfa**2))))
int_delta_tmp = assemble_scalar(form(delta_tmp*dx)) #form() l'ho scoperto per sbaglio. Senza esplode.
delta = delta_tmp/int_delta_tmp

# fluid definition
c0 = 340
rho_0 = 1.225
omega = Constant(msh, PETSc.ScalarType(1))
k0 = Constant(msh, PETSc.ScalarType(1))

#weak form
f = 1j*rho_0*omega*Q*delta
g = 1j*rho_0*omega/Z_s
a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx +  g * inner(u , v) * ds(5)
L = inner(f, v) * dx

# problem
uh = Function(V)
uh.name = "u"
problem = LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
p_mic = np.zeros((len(f_axis),1),dtype=complex)

# frequency for loop
print(" ")
for nf in range(0,len(f_axis)):
    print ("\033[A                             \033[A")
    freq = f_axis[nf]
    print("Computing frequency: " + str(freq) + " Hz...")  
    # Compute solution
    omega.value =f_axis[nf]*2*np.pi
    k0.value = 2*np.pi*f_axis[nf]/c0
    problem.solve()
        #Microphone pressure at specified point evaluation
    points = np.zeros((3,1))
    points[0][0] = mic[0]
    points[1][0] = mic[1]
    points[2][0] = mic[2]
    bb_tree = geometry.BoundingBoxTree(msh, msh.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = uh.eval(points_on_proc, cells)
    
    #building of sound pressure vector
    p_mic[nf] = u_values

    complex(0,1)
    
    # Define the complex number representing the imaginary unit
    imag_unit = complex(0, 1)


    # Access the real and imaginary parts of the function values
    uh_real_values = uh.vector[:].real
    uh_imag_values = uh.vector[:].imag
    
    # Time loop
        
    t0 = 0.0
    t1 = 0.5
    dt = 0.001
    num_time_steps = int((t1 - t0) / dt)
    file_name = "time_xdmf" + str(freq) + ".xdmf"
    # vtk_file = VTKFile(MPI.COMM_SELF, file_name, "w")
    xdmf_file = XDMFFile(MPI.COMM_SELF, os.path.join("3d_box_results", file_name), "w")
    
    xdmf_file.write_mesh(msh)

    
    for step in range(num_time_steps):
        t = t0 + step * dt

        # Compute the time-dependent factor exp(iωt)
        time_factor_value = np.exp(1j * omega.value * t)
        
        uh_time_function = Function(V)
        uh_time_function.vector[:] = uh.vector[:] * time_factor_value
        
        # Compute the time-dependent pressure field
        # uh_time = uh.vector[:] * time_factor_value
        xdmf_file.write_function(uh_time_function, t)

    print("Saved pressure field over time for: {} hz".format(freq))
    xdmf_file.close()
    
print ("\033[A                             \033[A")
print("JOB COMPLETED")

#creation of file [frequency Re(p) Im(p)]
Re_p = np.real(p_mic)
Im_p = np.imag(p_mic)
arr  = np.column_stack((f_axis,Re_p,Im_p))
print("Saving data to .csv file...")
np.savetxt("p_mic_3nd.csv", arr, delimiter=",")

#spectrum plot
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(f_axis, 20*np.log10(np.abs(p_mic)/2e-5), "k", linewidth=1, label="Sound pressure [dB]")
plt.grid(True)
plt.xlabel("frequency [Hz]")
plt.legend()
plt.show()

fig.savefig('test1234.png', dpi=fig.dpi)