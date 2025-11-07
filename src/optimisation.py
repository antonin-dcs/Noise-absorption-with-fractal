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
import compute_alpha

numpy.set_printoptions(threshold=numpy.inf)


def BelongsInteriorDomain(node):
	if (node < 0):
		return 1
	if node == 3:
		#print("Robin")
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
				#print(i+1,j, "-----", "i+1,j")
				chi[i + 1, j] = chi[i + 1, j] - mu * grad[i, j]
			if b == 2:
				#print(i - 1, j, "-----", "i - 1, j")
				chi[i - 1, j] = chi[i - 1, j] - mu * grad[i, j]
			if c == 2:
				#print(i, j + 1, "-----", "i , j + 1")
				chi[i, j + 1] = chi[i, j + 1] - mu * grad[i, j]
			if d == 2:
				#print(i, j - 1, "-----", "i , j - 1")
				chi[i, j - 1] = chi[i, j - 1] - mu * grad[i, j]

	return chi

def projection_star(arr):
    # Flatten the 2D array to 1D
    flat_arr = arr.flatten()
    m, n = arr.shape
    total_elements=0
    for i in range(0, m):
            for j in range(0, n):
                if domain_omega[i, j] == _env.NODE_ROBIN:
                    total_elements += 1
    beta=numpy.sum(arr)/total_elements
    # Calculate the target number of 1s
    k = round(beta * total_elements)
    
    # Sort indices by descending order of the values in flat_arr
    sorted_indices = numpy.argsort(-flat_arr)
    
    # Create a binary array with the required number of 1s and 0s
    binary_flat_arr = numpy.zeros(m*n, dtype=int)
    binary_flat_arr[sorted_indices[:k]] = 1
    
    # Reshape back to the original 2D shape
    binary_2d_arr = binary_flat_arr.reshape(m, n)
    
    return binary_2d_arr

def projection_star_on_robin(arr, domain_omega):
    """
    Binary projection that preserves volume (fraction of ones) but
    only acts on nodes marked as NODE_ROBIN in domain_omega.
    """
    M, N = arr.shape
    # boolean mask of robin nodes
    mask = (domain_omega == _env.NODE_ROBIN)
    robin_vals = arr[mask]

    if robin_vals.size == 0:
        return np.zeros_like(arr, dtype=int)

    # compute target number of ones (k) from average on robin nodes
    beta = robin_vals.sum() / robin_vals.size
    k = int(round(beta * robin_vals.size))

    # get indices of robin nodes sorted descending by value
    order = numpy.argsort(-robin_vals)  # indices into robin_vals

    # start with zeros, set top-k robin positions to 1
    result = numpy.zeros_like(arr, dtype=int)
    topk_indices = order[:k]
    # map flat robin positions back to 2D positions via mask
    robin_positions = numpy.argwhere(mask)  # array of shape (num_robin, 2)
    for idx in topk_indices:
        i, j = robin_positions[idx]
        result[i, j] = 1

    return result


def binary_projection(chi, beta):
    beta = numpy.clip(beta, 0.0, 1.0)
    chi_bin = numpy.zeros_like(chi, dtype=int)

    # Indices des lignes non nulles (où chi n'est pas totalement nulle)
    nonzero_rows = numpy.where(numpy.any(chi != 0, axis=1))[0]

    # Extraire uniquement les lignes utiles
    chi_nonzero = chi[nonzero_rows, :]
    chi_flat = chi_nonzero.flatten()
    n = len(chi_flat)

    if n == 0:
        # Si tout est nul, on retourne une matrice nulle
        return chi_bin

    # Projection binaire uniquement sur les lignes non nulles
    threshold_index = int(round(beta * n))
    idx_sorted = numpy.argsort(-chi_flat)
    chi_bin_flat = numpy.zeros_like(chi_flat, dtype=int)
    chi_bin_flat[idx_sorted[:threshold_index]] = 1

    # Reformater et replacer les lignes projetées à leur position d’origine
    chi_bin_nonzero = chi_bin_flat.reshape(chi_nonzero.shape)
    chi_bin[nonzero_rows, :] = chi_bin_nonzero

    return chi_bin

def your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj):
    """This function returns the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi.
    """
    #Parameters' initialisation
    k = 0
    (M, N) = numpy.shape(domain_omega)
    null=numpy.zeros((M,N))
    numb_iter = 100
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)
    while k < numb_iter and mu > 1e-10:
        print('---- iteration number = ', k)
        print('1. computing solution of Helmholtz problem, i.e., u')
        #u solution of the Helmholtz problem
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        print('2. computing solution of adjoint problem, i.e., p')
        #p solution of the adjoint problem
        p = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, -2*numpy.conjugate(u), null, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        print('3. computing objective function, i.e., energy')
        ene=energy[k]=your_compute_objective_function(domain_omega, u, spacestep)
        print('current energy:',ene)
        print('4. computing parametric gradient')
        #We set the new gradient on the robin's boundary
        grad=null.copy()
        for i in range(0, M):
            for j in range(0, N):
                if domain_omega[i, j] == _env.NODE_ROBIN:
                    grad[i,j]=-numpy.real(Alpha*u[i-1,j]*p[i-1,j])
        # While the energy is not diminishing
        while ene >= energy[k] and mu > 1e-10:
            #print('    a. computing gradient descent')
            chi_temp = chi - mu*grad
            # chi_temp=compute_gradient_descent(chi,grad,domain_omega,mu)
            #print('    b. computing projected gradient')
            chi_temp=compute_projected(chi_temp,domain_omega,V_obj)
            #print('    c. computing solution of Helmholtz problem, i.e., u')
            alpha_rob_temp=Alpha*chi_temp
            u=processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_temp)
            #print('    d. computing objective function, i.e., energy (E)')
            ene = your_compute_objective_function(domain_omega, u, spacestep)
            if ene<energy[k]:
                # The step is increased if the energy decreased
                mu = mu*1.1
                print('current energy good',ene)
            else:
                # The step is decreased if the energy increased
                mu = mu/2
                print('current energy bad',ene,'mu=',mu)
        #the chi we found is stored
        if ene<energy[k]:
            chi=chi_temp
            alpha_rob=alpha_rob_temp
        k += 1

    print('end. computing solution of Helmholtz problem, i.e., u')
    print('nb_iter',k)
    print('energy',energy[energy>0])
    #print('norm_grad_fin',numpy.sum(numpy.abs(grad)))

    return chi, energy, u, grad

def compute_projected(chi, domain, V_obj):
    """This function performs the projection of $\chi^n - mu*grad

    To perform the optimization, we use a projected gradient algorithm. This
    function caracterizes the projection of chi onto the admissible space
    (the space of $L^{infty}$ function which volume is equal to $V_{obj}$ and whose
    values are located between 0 and 1).

    :param chi: density matrix
    :param domain: domain of definition of the equations
    :param V_obj: characterizes the volume constraint
    :type chi: numpy.array((M,N), dtype=float64)
    :type domain: numpy.array((M,N), dtype=complex128)
    :type float: float
    :return:
    :rtype:
    """

    (M, N) = numpy.shape(domain)
    S = 0
    for i in range(M):
        for j in range(N):
            if domain[i, j] == _env.NODE_ROBIN:
                S = S + 1

    B = chi.copy()
    l = 0
    chi = preprocessing.set2zero(chi, domain)

    V = numpy.sum(numpy.sum(chi)) / S
    debut = -numpy.max(chi)
    fin = numpy.max(chi)
    ecart = fin - debut
    # We use dichotomy to find a constant such that chi^{n+1}=max(0,min(chi^{n}+l,1)) is an element of the admissible space
    while ecart > 10 **-6:
        # calcul du milieu
        l = (debut + fin) / 2
        for i in range(M):
            for j in range(N):
                chi[i, j] = numpy.maximum(0, numpy.minimum(B[i, j] + l, 1))
        chi = preprocessing.set2zero(chi, domain)
        V = sum(sum(chi)) / S
        if V > V_obj:
            fin = l
        else:
            debut = l
        ecart = fin - debut
        # print('le volume est', V, 'le volume objectif est', V_obj)

    return chi


def your_compute_objective_function(domain_omega, u, spacestep):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 + mu1*(Vol(domain_omega)-V_0)
    """
    return numpy.sum(numpy.abs(u)**2)* (spacestep**2)




if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Feel free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 2 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    if level == 0: freq = 973.7373737373737
    elif level == 1: freq = 802.0202020202021
    elif level == 2: freq = 1089.8989898989898
    elif level == 3: freq = 800
    else : freq = 1000

    # -- set parameters of the partial differential equation
    speed = 343.0        # speed of sound (m/s) — choose one value and keep it everywhere
    omega = 2.0 * numpy.pi * freq
    wavenumber = omega / speed



    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)


    def g(y):
        sigma=1.0
        y0=0.0
        return 10*numpy.exp(-(y-y0)**2/2*sigma**2)/(sigma*numpy.sqrt(2*numpy.pi))
    

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0
    g_1=numpy.vectorize(lambda y: g(y))
    g_dir = numpy.zeros((M,N))
    g_dir[0,:] = g_1(numpy.arange(N))

    # spherical wave defined on top
    #f_dir[:, :] = 0.0
    #f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix

    #chi = preprocessing.set_chi_partitioned(M, N, x, y, 3, 6)
    chi = preprocessing._set_chi(M, N, x, y, 0.34)

    chi = preprocessing.set2zero(chi, domain_omega)

    # -- define absorbing material
    #Alpha = 10.0 - 10.0 * 1j
    # -- this is the function you have written during your project
    #Alpha = compute_alpha.real_to_complex(compute_alpha.compute_alpha(omega,'POLYESTER'))
    Alpha, _ = compute_alpha.compute_alpha(omega, 'POLYESTER')
    
    chi = compute_projected(chi, domain_omega, 0.34)

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
    print("volume", V_obj)
    mu = 5  # initial gradient step
    
    alpha_rob = Alpha * chi

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, g_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        
    chi0 = chi.copy()
    u0 = u.copy()
    

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    energy = numpy.zeros((100+1, 1), dtype=numpy.float64)
    #chi, energy, u, grad = your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
    #                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                       Alpha, mu, chi, V_obj, mu1, V_0)
    chi, energy, u, grad = your_optimization_procedure(domain_omega, spacestep, wavenumber, f, g_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                        Alpha, mu, chi, V_obj)
    # --- en of optimization

    chin = chi.copy()
    un = u.copy()
        # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)

    chibinary = projection_star(chin)
    #chibinary=projection_star_on_robin(chin, domain_omega)
    postprocessing._plot_controled_binary_solution(un, chibinary)


    err = un - u0
    postprocessing._plot_error(err)
     
    postprocessing._plot_energy_history(energy)

    print('End.')


import matplotlib.pyplot as plt
import numpy as np


def profil_chi_steps():
    # Suppose que gamma correspond à la coordonnée x
    gamma = np.linspace(0, 50, chi.shape[1])  # axe des x (normalisé de 0 à 1)
    chi_line = chin[chin.shape[0] // 2, :]   # ligne centrale (en hauteur)
    chi0_line = chi0[chi0.shape[0] // 2, :]

    plt.figure(figsize=(8, 5))
    plt.plot(gamma, chi_line, drawstyle='steps-mid')
    plt.plot(gamma, chi0_line, drawstyle='steps-mid')
    plt.xlabel("Gamma")
    plt.ylabel("Chi")
    plt.title("Profil de Chi en fonction de Gamma")
    plt.grid(True)
    plt.show()



def plot_energies():
    c = 343.0  # vitesse du son (m/s)
    frequencies = np.linspace(600, 1100, 100)  # fréquences en Hz

    energies_chi0 = []
    energies_chin = []
    energies_with_porous = []

    for freq in frequencies:
        print(f"Résolution à {freq:.1f} Hz")
        wavenumber = 2 * np.pi * freq / c

        # Cas 1 : avant optimization chi=chi0
        alpha_rob_0 = Alpha * chi0
        u0_freq = processing.solve_helmholtz(domain_omega, spacestep, wavenumber,
                                            f, g_dir, f_neu, f_rob,
                                            beta_pde, alpha_pde, alpha_dir,
                                            beta_neu, beta_rob, alpha_rob_0)
        E0 = your_compute_objective_function(domain_omega, u0_freq, spacestep)
        energies_chi0.append(E0)

        # Cas 2 : avec matériau optimisé (chi = chin)
        alpha_rob_opt = Alpha * chibinary
        un_freq = processing.solve_helmholtz(domain_omega, spacestep, wavenumber,
                                            f, g_dir, f_neu, f_rob,
                                            beta_pde, alpha_pde, alpha_dir,
                                            beta_neu, beta_rob, alpha_rob_opt)
        En = your_compute_objective_function(domain_omega, un_freq, spacestep)
        energies_chin.append(En)

        # chi = 1

        chi1 = numpy.ones_like(domain_omega)
        chi1 = preprocessing.set2zero(chi1, domain_omega)  # seulement sur frontière Robin

        alpha_rob1 = Alpha * chi1

        u1 = processing.solve_helmholtz(
            domain_omega, spacestep, wavenumber,
            f, g_dir, f_neu, f_rob,
            beta_pde, alpha_pde, alpha_dir,
            beta_neu, beta_rob, alpha_rob1
        )

        E1 = your_compute_objective_function(domain_omega, u1, spacestep)
        energies_with_porous.append(E1)

    # Conversion en arrays
    energies_chi0 = np.array(energies_chi0)
    energies_chin = np.array(energies_chin)
    energies_with_porous = np.array(energies_with_porous)

    maximum = np.argmax(energies_chi0)
    print("maximum :" , frequencies[maximum])

    # Tracé
    plt.figure(figsize=(9, 6))
    plt.plot(frequencies, energies_chi0, 'o-', label="Avant optimisation")
    plt.plot(frequencies, energies_chin, 's-', label="Après optimisation")
    plt.plot(frequencies, energies_with_porous, 's-', label="chi = 1")
    plt.title("Énergie acoustique $E(f)$ en fonction de la fréquence")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Énergie acoustique $E(f)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


#profil_chi_steps()
plot_energies()