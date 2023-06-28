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
# gmsh.initialize()

# %%

from dolfinx.io import gmshio
msh, cell_tags, facet_tags = gmshio.read_from_msh("mesh3D.msh", MPI.COMM_SELF, 0, gdim=3)
# %%

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

# %%

# mic cell_marker
Sx = 0.0
Sy = 0.0
Sz = 0.0
# %%
#frequency range definition
f_axis = np.arange(50, 2005, 5)

#Mic position definition
mic = np.array([0.0, 0.0, 0.0])

deg = 3
# %%
# Test and trial function space
V = FunctionSpace(msh, ("CG", 2))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = Function(V)

# %%
# Source amplitude
Q = 0.0001

#Source definition position = (Sx,Sy)
Sx = 0.0
Sy = 0.0
Sz = 0.0

#Narrow normalized gauss distribution
alfa = 0.015
delta_tmp = Function(V)
# delta_tmp.interpolate(lambda x : 1/(np.abs(alfa)*np.sqrt(np.pi))*np.exp(-(((x[0]-Sx)**2+(x[1]-Sy)**2)/(alfa**2))))
delta_tmp.interpolate(lambda x: 1 / (np.abs(alfa) * np.sqrt(np.pi)) * np.exp(-(((x[0] - Sx) ** 2 + (x[1] - Sy) ** 2 + (x[2] - Sz) ** 2) / (alfa ** 2))))
# delta_tmp.interpolate(lambda x: 1 / (np.abs(alfa) * (2 * np.pi) ** (3/2)) * np.exp(-(((x[0] - Sx) ** 2 + (x[1] - Sy) ** 2 + (x[2] - Sz) ** 2) / (2 * alfa ** 2))))

int_delta_tmp = assemble_scalar(form(delta_tmp*dx)) #form() l'ho scoperto per sbaglio. Senza esplode.
delta = delta_tmp/int_delta_tmp
# %%
# fluid definition
c0 = 340
rho_0 = 1.225
omega = Constant(msh, PETSc.ScalarType(1))
k0 = Constant(msh, PETSc.ScalarType(1))

ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

print(facet_tags)

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
print("solving..")
omega.value = omega.value =440*2*np.pi
k0.value = 2*np.pi*440/c0

problem.solve()