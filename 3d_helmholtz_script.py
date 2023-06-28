# %%
import numpy as np
import os
import ufl
import dolfinx
from dolfinx import geometry
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form, Constant, Expression
from dolfinx import geometry
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, VTKFile
from dolfinx.mesh import create_unit_square
from ufl import dx, grad, inner
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import gmshio
import matplotlib.pyplot as plt

# %%
import gmsh

# Initialize Gmsh
gmsh.initialize()

# %%
#from dolfinx.io import gmshio

#path = os.path.join("results", "room_mesh_test.msh")
#msh, cell_tags, facet_tags = gmshio.read_from_msh("results/" +"room_mesh_test.msh", MPI.COMM_SELF, 0, gdim=3)


# %%
# Choose if Gmsh output is verbose
# Set the desired mesh size
mesh_size = 0.017  # Replace with your desired mesh size

# Initialize the Gmsh model
gmsh.initialize()
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
gmsh.option.setNumber("General.Terminal", 0)

model = gmsh.model()
model.add("room")
model.setCurrent("room")
stl_path = os.path.join("results", "box.step")
room = gmsh.model.occ.importShapes(stl_path, highestDimOnly=False)



# Synchronize OpenCascade representation with gmsh model
model.occ.synchronize()

# Add physical marker for cells. It is important to call this function
# after OpenCascade synchronization

# Generate the mesh

characteristic_length = 0.017  # Desired mesh size
gmsh.model.mesh.setSize(gmsh.model.getEntities(2), characteristic_length)
model.mesh.generate(dim=3)
# model.add_physical_group(dim=3, tags=room)
model.addPhysicalGroup(3, [1])

# Create a DOLFINx mesh (same mesh on each rank)
msh, cell_markers, facet_markers = gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
msh.name = "room"
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

gmsh.finalize()

# %%


# %%
print(facet_markers.values)

# %%
# mic cell_markerstion
Sx = 0.4
Sy = -0.3
Sz = 0.1

# %%
#frequency range definition
f_axis = np.arange(50, 2005, 5)

#Mic position definition
mic = np.array([0.3, 0.3, 0])

deg = 3

# %%
# Test and trial function space
V = FunctionSpace(msh, ("CG", deg))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = Function(V)

# %%
# Source amplitude
Q = 0.0001

#Source definition position = (Sx,Sy)
Sx = 0.4
Sy = -0.3
Sz = 0.1

#Narrow normalized gauss distribution
alfa = 0.015
delta_tmp = Function(V)
# delta_tmp.interpolate(lambda x : 1/(np.abs(alfa)*np.sqrt(np.pi))*np.exp(-(((x[0]-Sx)**2+(x[1]-Sy)**2)/(alfa**2))))
delta_tmp.interpolate(lambda x: 1 / (np.abs(alfa) * np.sqrt(np.pi)) * np.exp(-(((x[0] - Sx) ** 2 + (x[1] - Sy) ** 2 + (x[2] - Sz) ** 2) / (alfa ** 2))))

int_delta_tmp = assemble_scalar(form(delta_tmp*dx)) #form() l'ho scoperto per sbaglio. Senza esplode.
delta = delta_tmp/int_delta_tmp

# %%
# fluid definition
c0 = 340
rho_0 = 1.225
omega = Constant(msh, PETSc.ScalarType(1))
k0 = Constant(msh, PETSc.ScalarType(1))

ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers)

# %%
Z_s = Constant(msh, PETSc.ScalarType(1))

#weak form
f = 1j*rho_0*omega*Q*delta
g = 1j*rho_0*omega/Z_s
a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx +  g * inner(u , v) * ds(5)
L = inner(f, v) * dx

# problem
uh = Function(V)
uh.name = "u"
problem = LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# %%
print(problem)

# %%
omega.value = omega.value =440*2*np.pi
k0.value = 2*np.pi*440/c0

problem.solve()

# %%
print("starrting..")

omega.value =440*2*np.pi
k0.value = 2*np.pi*440/c0
print("solving problem")
problem.solve()
print("solved")
t0 = 0.0
t1 = 0.5
dt = 0.01


num_time_steps = int((t1 - t0) / dt)
file_name = "440_hz_box_mesh"+ ".xdmf"
# vtk_file = VTKFile(MPI.COMM_SELF, file_name, "w")
subfolder = "results"
if not os.path.exists(subfolder):
    os.makedirs(subfolder)
xdmf_file = XDMFFile(MPI.COMM_SELF, os.path.join(subfolder, file_name), "w")

xdmf_file.write_mesh(msh)


for step in range(num_time_steps):
    t = t0 + step * dt
    print("calculating time step: ", t)

    # Compute the time-dependent factor exp(iωt)
    time_factor_value = np.exp(1j * omega.value * t)

    uh_time_function = Function(V)
    uh_time_function.vector[:] = uh.vector[:] * time_factor_value

    # Compute the time-dependent pressure field
    # uh_time = uh.vector[:] * time_factor_value
    xdmf_file.write_function(uh_time_function, t)

# %%
p_mic = np.zeros((len(f_axis),1),dtype=complex)
# frequency for loop

subfolder = "results"
if not os.path.exists(subfolder):
    os.makedirs(subfolder)
print(" ")
for nf in range(0,len(f_axis)):
    print ("\033[A                             \033[A")
    freq = f_axis[nf]
    print("Computing frequency: " + str(freq) + " Hz...")  

    # Compute solution
    omega.value =f_axis[nf]*2*np.pi
    k0.value = 2*np.pi*f_axis[nf]/c0
    problem.solve()
    
    
    complex(0,1)
    
    # Define the complex number representing the imaginary unit
    imag_unit = complex(0, 1)


    # Access the real and imaginary parts of the function values
    uh_real_values = uh.vector[:].real
    uh_imag_values = uh.vector[:].imag
    
    # Time loop
        
    t0 = 0.0
    t1 = 0.5
    dt = 0.01
    num_time_steps = int((t1 - t0) / dt)
    file_name = "time_xdmf" + str(freq) + ".xdmf"
    # vtk_file = VTKFile(MPI.COMM_SELF, file_name, "w")
    xdmf_file = XDMFFile(MPI.COMM_SELF, os.path.join(subfolder, file_name), "w")
    
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

        
        # xdmf_file.write_function(uh_time, step)
    


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
    
print ("\033[A                             \033[A")
print("JOB COMPLETED")

# %%



