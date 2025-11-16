#"""Optimization utilities for acoustic absorption (projected gradient).
"""
This module contains routines to optimize a spatial absorption density `chi`
on the Robin boundary of a discrete domain for a Helmholtz PDE. It provides
low-level helpers (gradient update, binary projection) and a high-level
optimizer that solves forward and adjoint Helmholtz problems to compute
gradients and enforce a volume constraint via projection.

Main functions:
- compute_gradient_descent: local gradient update near Robin nodes
- compute_projected: project chi into [0,1] with a prescribed volume
- projection_star / projection_star_on_robin / binary_projection: binary
    projection utilities
- your_optimization_procedure: main projected-gradient optimization loop
- your_compute_objective_function: compute acoustic energy objective

Notes:
- Operates on numpy arrays and relies on global `_env.NODE_ROBIN` and
    helpers in `preprocessing`, `processing`, and `compute_alpha`.
"""

# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os


# MRG packages
import _env
import preprocessing
import processing
import postprocessing
import solutions
#from gradient_descent import compute_gradient_descent
from projected_chi import compute_projected, compute_binary
import compute_alpha

def your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj, mu1, V_0):

    k = 0
    (M, N) = numpy.shape(domain_omega)
    null = numpy.zeros((M,N))
    numb_iter = 100
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)

    while k < numb_iter and mu > 10**(-5):
        print('---- iteration number = ', k)
        print('1. computing solution of Helmholtz problem, i.e., u')
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)

        print('2. computing solution of adjoint problem, i.e., p')
        p = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, -2*numpy.conjugate(u), null, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)

        print('3. computing objective function, i.e., energy')
        ene = energy[k] = your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0)

        print('4. computing parametric gradient')
        grad = numpy.zeros_like(chi, dtype=float)
        for i in range(M):
            for j in range(N):
                if domain_omega[i, j] == _env.NODE_ROBIN:
                    grad[i, j] = - numpy.real(Alpha * u[i-1, j] * p[i-1, j])


        # tentative update
        chi_temp = chi - mu * grad
        chi_temp = compute_projected(chi_temp, domain_omega, V_obj)
        alpha_rob = Alpha * chi_temp


        # recompute energy
        u_temp = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        ene_new = your_compute_objective_function(domain_omega, u_temp, spacestep, mu1, V_0)

        # adapt step
        if ene_new < ene:
            chi = chi_temp
            mu = mu * 1.1
            energy[k+1] = ene_new
        else:
            mu = mu / 2

        k += 1

    print('end. computing solution of Helmholtz problem, i.e., u')

    return chi, energy, u, grad


def optimization_multifrequence(domain_omega, spacestep, freq_list, f, f_dir, f_neu, f_rob,
                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
                       mu, chi, V_obj, mu1, V_0):
    """
    Multi-frequency optimization:
    The energy considered is the sum of the energies for all frequencies in freq_list.
    """
    k = 0
    (M, N) = numpy.shape(domain_omega)
    null = numpy.zeros((M, N))
    numb_iter = 100
    energy = numpy.zeros((numb_iter + 1, 1), dtype=numpy.float64)

    speed = 343.0
    print("---- Multi-frequency optimization ----")
    print(f"Frequencies = {freq_list}")

    # initialize u (for return)
    u = numpy.zeros((M, N), dtype=complex)

    while k < numb_iter and mu > 1e-5:
        print(f"\n---- iteration number = {k}")
        total_energy = 0.0
        grad = numpy.zeros_like(chi, dtype=float)

        # Loop over all frequencies
        for freq in freq_list:
            omega = 2.0 * numpy.pi * freq
            wavenumber = omega / speed
            Alpha, _ = compute_alpha.compute_alpha(omega, 'POLYESTER')
            alpha_rob = Alpha * chi

            # Solve direct problem
            u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber,
                                           f, f_dir, f_neu, f_rob,
                                           beta_pde, alpha_pde, alpha_dir,
                                           beta_neu, beta_rob, alpha_rob)

            # Solve adjoint problem
            p = processing.solve_helmholtz(domain_omega, spacestep, wavenumber,
                                           -2 * numpy.conjugate(u), null, f_neu, f_rob,
                                           beta_pde, alpha_pde, alpha_dir,
                                           beta_neu, beta_rob, alpha_rob)

            # Energy contribution from this frequency
            total_energy += your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0)

            # Gradient contribution
            for i in range(1, M):  # start at 1 because of i-1 indexing
                for j in range(N):
                    if domain_omega[i, j] == _env.NODE_ROBIN:
                        grad[i, j] += -numpy.real(Alpha * u[i-1, j] * p[i-1, j])

        energy[k] = total_energy
        print(f"Total energy = {total_energy:.6e}")

        # Tentative update
        chi_temp = chi - mu * grad
        chi_temp = compute_projected(chi_temp, domain_omega, V_obj)

        # Compute energy with updated chi
        ene_new = 0.0
        u_temp = numpy.zeros_like(u, dtype=complex)
        for freq in freq_list:
            omega = 2.0 * numpy.pi * freq
            wavenumber = omega / speed
            Alpha, _ = compute_alpha.compute_alpha(omega, 'POLYESTER')
            alpha_rob_temp = Alpha * chi_temp
            u_temp = processing.solve_helmholtz(domain_omega, spacestep, wavenumber,
                                                f, f_dir, f_neu, f_rob,
                                                beta_pde, alpha_pde, alpha_dir,
                                                beta_neu, beta_rob, alpha_rob_temp)
            ene_new += your_compute_objective_function(domain_omega, u_temp, spacestep, mu1, V_0)

        # Adapt step size
        if ene_new < total_energy:
            chi = chi_temp
            mu = mu * 1.1
            u = u_temp  # store last computed field for return
            energy[k + 1] = ene_new
            print(f"Energy decreased → accepted step (E={ene_new:.6e}, mu={mu:.3e})")
        else:
            mu = mu / 2
            print(f"Energy increased → reducing step (mu={mu:.3e})")

        k += 1

    print("\nEnd of multi-frequency optimization.")
    print(f"Number of iterations: {k}")
    print("Final energy values:", energy[energy > 0].flatten())
    return chi, energy, u, grad


def your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 + mu1*(Vol(domain_omega)-V_0)

    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation;
        mu1: float, it is the constant that defines the importance of the volume
        constraint;
        V_0: float, it is a reference volume.
    """

    return numpy.sum(numpy.abs(u)**2)* (spacestep**2)



def BelongsInteriorDomain(node):
	if (node < 0):
		return 1
	if node == 3:
		print("Robin")
		return 2
	else:
		return 0


def compute_gradient_descent(chi, grad, domain, mu):
	"""This function makes the gradient descent.
	This function has to be used before the 'Projected' function that will project
	the new element onto the admissible space.
	:param chi: density of absorption define everywhere in the domain
	:param grad: parametric gradient associated to the problem
	:param domain: domain of definition of the equations
	:param mu: step of the descent
	:type chi: numpy.array((M,N), dtype=float64
	:type grad: numpy.array((M,N), dtype=float64)
	:type domain: numpy.array((M,N), dtype=int64)
	:type mu: float
	:return chi:
	:rtype chi: numpy.array((M,N), dtype=float64

	.. warnings also: It is important that the conditions be expressed with an "if",
			not with an "elif", as some points are neighbours to multiple points
			of the Robin frontier.
	"""

	(M, N) = numpy.shape(domain)
	# for i in range(0, M):
	# 	for j in range(0, N):
	# 		if domain_omega[i, j] != _env.NODE_ROBIN:
	# 			chi[i, j] = chi[i, j] - mu * grad[i, j]
	# # for i in range(0, M):
	# 	for j in range(0, N):
	# 		if preprocessing.is_on_boundary(domain[i , j]) == 'BOUNDARY':
	# 			chi[i,j] = chi[i,j] - mu*grad[i,j]
	# print(domain,'jesuisla')
	#chi[50,:] = chi[50,:] - mu*grad[50,:]
	for i in range(1, M - 1):
		for j in range(1, N - 1):
			#print(i,j)
			#chi[i,j] = chi[i,j] - mu * grad[i,j]
			a = BelongsInteriorDomain(domain[i + 1, j])
			b = BelongsInteriorDomain(domain[i - 1, j])
			c = BelongsInteriorDomain(domain[i, j + 1])
			d = BelongsInteriorDomain(domain[i, j - 1])
			if a == 2:
				print(i+1,j, "-----", "i+1,j")
				chi[i + 1, j] = chi[i + 1, j] - mu * grad[i, j]
			if b == 2:
				print(i - 1, j, "-----", "i - 1, j")
				chi[i - 1, j] = chi[i - 1, j] - mu * grad[i, j]
			if c == 2:
				print(i, j + 1, "-----", "i , j + 1")
				chi[i, j + 1] = chi[i, j + 1] - mu * grad[i, j]
			if d == 2:
				print(i, j - 1, "-----", "i , j - 1")
				chi[i, j - 1] = chi[i, j - 1] - mu * grad[i, j]

	return chi

if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 2 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    freq = 812
    speed = 343.0        # speed of sound (m/s) 
    omega = 2.0 * numpy.pi * freq
    wavenumber = omega / speed
    plage_freq1 = [645, 812]
    plage_freq2 = [645, 700, 812, 1020]

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, 0:N] = 1.0
    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    def g(y):
        sigma = 1.0
        y0 = 0.0  # center of the domain (adjust if needed)
        y_array = numpy.asarray(y)
        return 10 * numpy.exp(-(y_array-y0)**2 / (2 * sigma**2)) / (sigma * numpy.sqrt(2 * numpy.pi))

    #gaussian source
    f_dir[:, :] = 0.0
    f_dir[0, :] = g(x[:N])

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y, 0.4)
    chi = preprocessing.set2zero(chi, domain_omega)

    # -- define absorbing material
    #Alpha = 10.0 - 10.0 * 1j
    Alpha, _ = compute_alpha.compute_alpha(omega, 'POLYESTER')
    alpha_rob = Alpha * chi

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
    mu = 5  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional


    #verify the source shape
    import matplotlib.pyplot as plt
    plt.plot(x[:N], f_dir[0, :])
    plt.title("Source gaussienne considérée")
    plt.xlabel("y")
    plt.ylabel("g(y)")
    plt.show()

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()


    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    
    chi, energy, u, grad = your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                        Alpha, mu, chi0, V_obj, mu1, V_0)
                        

    # --- en of optimization

    chin = chi.copy()
    un = u.copy()

    # with multiple frequencies
    chi, energy, u, grad = optimization_multifrequence(domain_omega, spacestep, plage_freq1, f, f_dir, f_neu, f_rob,
                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob,
                       mu, chi0, V_obj, mu1, V_0)
    
    chi_multiple = chi.copy()
    u_multiple = u.copy()
    
    # compute beta as average chi on Robin nodes
    #beta = numpy.sum(chi0[domain_omega == _env.NODE_ROBIN]) / S
    chibinary = compute_binary(chi_multiple, domain_omega)
    postprocessing._plot_controled_binary_solution(u_multiple, chibinary)
    postprocessing.plot_module(u0, chi0)
    postprocessing.plot_module_controled(u_multiple, chi_multiple)


    # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(u_multiple, chi_multiple)
    err = u_multiple - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    print('End.')


def energy_f_freq(f_min, f_max, g, chi0, label, chi, domain_omega, x, spacestep,
                  beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, f, f_neu, f_rob,
                  nb_pts=100, proj=False, beta=0.4):
    """
    Compute the evolution of the energy as a function of frequency
    for a given material distribution chi and source function g(y).

    Parameters
    ----------
    f_min, f_max : float
        Frequency range (Hz)
    g : function
        Source function g(y)
    chi : ndarray
        Material density
    domain_omega : ndarray
        Domain mesh
    x : ndarray
        Spatial coordinates along y-axis
    spacestep : float
        Grid spacing
    PDE coefficients : various
    nb_pts : int
        Number of frequency points
    proj : bool
        Whether to use projected chi
    beta : float
        Volume fraction (for plot label)
    """

    (M, N) = domain_omega.shape

    # Project chi if requested
    if proj:
        chi_projected = compute_binary(chi, domain_omega)

    # Frequency array
    freqs = numpy.linspace(f_min, f_max, nb_pts)
    omegas = 2 * numpy.pi * freqs

    energies0 = numpy.zeros(nb_pts)
    energies = numpy.zeros(nb_pts)
    energies_projected = numpy.zeros(nb_pts) if proj else None

    for k, omega_k in enumerate(omegas):

        # Gaussian Dirichlet source at top boundary
        g_dir = numpy.zeros((M, N), dtype=float)
        g_dir[0, :] = g(x[:N])  # ensure source matches grid length

        # Compute absorption coefficient
        Alpha, _ = compute_alpha.compute_alpha(omega_k, 'POLYESTER')

        # Solve Helmholtz PDE
        u0 = processing.solve_helmholtz(domain_omega, spacestep, omega_k/343.0,
                                       f, g_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir,
                                       beta_neu, beta_rob, Alpha * chi0)    

        u = processing.solve_helmholtz(domain_omega, spacestep, omega_k/343.0,
                                       f, g_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir,
                                       beta_neu, beta_rob, Alpha * chi)

        # Compute energy
        energies[k] = your_compute_objective_function(domain_omega, u, spacestep, 0.0, 0.0)
        energies0[k] = your_compute_objective_function(domain_omega, u0, spacestep, 0.0, 0.0)

        # Projected version if requested
        if proj:
            u_proj = processing.solve_helmholtz(domain_omega, spacestep, omega_k/343.0,
                                                f, g_dir, f_neu, f_rob,
                                                beta_pde, alpha_pde, alpha_dir,
                                                beta_neu, beta_rob, Alpha * chi_projected)
            energies_projected[k] = your_compute_objective_function(domain_omega, u_proj, spacestep, 0.0, 0.0)

    # Plot results
    import matplotlib.pyplot as plt
    plt.plot(freqs, energies0, marker='.', label=label)
    #plt.plot(freqs, energies, marker='.', linestyle='-', label='non-projeté')
    if proj:
        plt.plot(freqs, energies_projected, marker='.', linestyle='-', label="Projeté (optimisé sur 2 fréquences)")
    plt.legend()
    plt.xlabel('Fréquence [Hz]')
    plt.ylabel('Énergie')
    plt.title(r"Évolution de l'énergie pour $\beta$=" + str(beta))
    plt.grid(True)
    plt.show()


chin_proj = compute_binary(chin, domain_omega)
energy_f_freq(600, 1100, g, chin_proj, "Projeté (optimisé sur 1 fréquence)", chi_multiple, domain_omega, x, spacestep,
                  beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, f, f_neu, f_rob,
                  nb_pts=200, proj=True, beta=0.4)



chi1 = numpy.ones_like(domain_omega)
chi1 = preprocessing.set2zero(chi1, domain_omega)

#energy_f_freq(600, 1100, g, chi1, "Matériau absorbant partout", chi_multiple, domain_omega, x, spacestep,
 #                 beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, f, f_neu, f_rob,
  #                nb_pts=200, proj=True, beta=0.4)


def profil_chi_steps():
    # Define the gamma axis (x-coordinate)
    gamma = numpy.linspace(0, chin.shape[1] - 1, chin.shape[1])
    
    # Take the middle horizontal slice of each chi
    chi_line = chi_multiple[chi_multiple.shape[0] // 2, :]
    chi0_line = chi0[chi0.shape[0] // 2, :]
    chibinary_line = chibinary[chibinary.shape[0] // 2, :]

    # Plot
    plt.figure(figsize=(8, 5))
    #plt.plot(gamma, chi0_line, drawstyle='steps-mid', label='Initial χ₀')
    plt.plot(gamma, chi_line, label='Optimized χ')
    plt.plot(gamma, chibinary_line, drawstyle='steps-mid', label='Projected χ (binary)')
    plt.xlabel("Γ (gamma coordinate)")
    plt.ylabel("χ value")
    plt.title("Profile of χ as a function of Γ (before and after projection)")
    plt.grid(True)
    plt.legend()
    plt.show()


profil_chi_steps()