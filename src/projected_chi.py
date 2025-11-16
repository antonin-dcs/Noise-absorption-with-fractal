import numpy
import _env
from preprocessing import set2zero


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
    chi = set2zero(chi, domain)

    V = numpy.sum(numpy.sum(chi)) / S
    debut = -numpy.max(chi)
    fin = numpy.max(chi)
    ecart = fin - debut
    # We use dichotomy to find a constant such that chi^{n+1}=max(0,min(chi^{n}+l,1)) is an element of the admissible space
    while ecart > 10 ** -4:
        # calcul du milieu
        l = (debut + fin) / 2
        for i in range(M):
            for j in range(N):
                chi[i, j] = numpy.maximum(0, numpy.minimum(B[i, j] + l, 1))
        chi = set2zero(chi, domain)
        V = sum(sum(chi)) / S
        if V > V_obj:
            fin = l
        else:
            debut = l
        ecart = fin - debut
        # print('le volume est', V, 'le volume objectif est', V_obj)

    return chi


def compute_binary(chi, domain_omega):
    # Flatten the 2D array to 1D
    flat_arr = chi.flatten()
    m, n = chi.shape
    total_elements=0
    for i in range(0, m):
            for j in range(0, n):
                if domain_omega[i, j] == _env.NODE_ROBIN:
                    total_elements += 1
    beta=numpy.sum(chi)/total_elements
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
