# %%
import gmsh
gmsh.initialize()

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
# approximation space polynomial degree
deg = 2

# number of elements in each direction of msh
n_elem = 64

msh = create_unit_square(MPI.COMM_SELF, n_elem, n_elem)
n = ufl.FacetNormal(msh)

# %%
#frequency range definition
f_axis = np.arange(50, 2005, 5)

#Mic position definition
mic = np.array([0.3, 0.3, 0])

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
Sx = 0.5
Sy = 0.5

#Narrow normalized gauss distribution
alfa = 0.015
delta_tmp = Function(V)
delta_tmp.interpolate(lambda x : 1/(np.abs(alfa)*np.sqrt(np.pi))*np.exp(-(((x[0]-Sx)**2+(x[1]-Sy)**2)/(alfa**2))))
int_delta_tmp = assemble_scalar(form(delta_tmp*dx)) #form() l'ho scoperto per sbaglio. Senza esplode.
delta = delta_tmp/int_delta_tmp

# %%
# fluid definition
c0 = 340
rho_0 = 1.225
omega = Constant(msh, PETSc.ScalarType(1))
k0 = Constant(msh, PETSc.ScalarType(1))

# %%
#building the weak form
f = 1j*rho_0*omega*Q*delta
a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx
L = inner(f, v) * dx

# %%
#building the problem
uh = Function(V)
uh.name = "u"
problem = LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

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
    
    
    """ # Time loop
        
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
     """


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
#creation of file [frequency Re(p) Im(p)]
Re_p = np.real(p_mic)
Im_p = np.imag(p_mic)
arr  = np.column_stack((f_axis,Re_p,Im_p))
print("Saving data to .csv file...")
np.savetxt("p_mic_2nd.csv", arr, delimiter=",")

# %%
#spectrum plot
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(f_axis, 20*np.log10(np.abs(p_mic)/2e-5), "k", linewidth=1, label="Sound pressure [dB]")
plt.grid(True)
plt.xlabel("frequency [Hz]")
plt.legend()
plt.show()

# %%



