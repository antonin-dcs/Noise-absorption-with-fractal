"""
compute_alpha.py
----------------

This module computes the optimal acoustic parameter α(ω) for a set of frequencies.
For each frequency ω, the script uses a BFGS optimization algorithm to minimize
the global modeling error between:

- the measured acoustic response of a porous material, and
- the simplified analytical model used in the simulation.

In other words, α(ω) is chosen so that the simplified model reproduces as closely
as possible the real acoustic behavior of the material across multiple materials
or experimental datasets.

This parameter identification step is required before running the full
Helmholtz-based wave propagation simulations, as it ensures that the material
model remains physically meaningful and correctly calibrated.
"""

# Python packages
import matplotlib.pyplot
import numpy
import scipy
from scipy.optimize import minimize
import scipy.io
from scipy.integrate import quad_vec


def real_to_complex(z):
    return z[0] + 1j * z[1]


def complex_to_real(z):
    return numpy.array([numpy.real(z), numpy.imag(z)])


class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        # .. todo: deepcopy here if returning objects
        return self.memo[args]


def compute_alpha(omega, material):
    """
    .. warning: $w = 2 \pi f$
    w is called circular frequency
    f is called frequency
    """
    if material=='GLASS WOOL':
        # parameters of the material
        phi = 0.9999
        gamma_p = 7.0 / 5.0
        sigma = 38863.0
        rho_0 = 16.0
        alpha_h = 1.0
        c_0 = 340.0

    #source : https://www.researchgate.net/publication/281407464_EXPERIMENTAL_CHARACTERIZATION_OF_MATERIALS_FOR_ACOUSTIC_PERFORMANCE_WITH_APPLICATIONS

    if material=='MELAMINE FOAM':
        # parameters of the material
        phi = 0.99
        gamma_p = 7.0 / 5.0
        sigma = 14000.0
        rho_0 = 8.3
        alpha_h = 1.02
        c_0 = 340.0

    if material=='Polyurethane foam':
        # parameters of the material
        phi = 0.98  # porosity
        gamma_p = 7.0 / 5.0
        sigma = 45000.0  # resitivity
        rho_0 = 30
        alpha_h = 2.01  # tortuosity 
        c_0 = 340.0     # sound speed in air m/s

    #https://www.researchgate.net/publication/261324486_Experimental_study_of_wood_acoustic_absorption_characteristics/figures?lo=1

    if material=='PINELT':
        # parameters of the material
        phi = 0.66
        gamma_p = 7.0 / 5.0
        sigma = 37316.0
        rho_0 = 1.2
        alpha_h = 1.23
        c_0 = 340.0

    if material=='SAPELILT':
        phi = 0.509  # porosity
        gamma_p = 7.0 / 5.0
        sigma = 214855.0  # resitivity
        rho_0 = 1.2
        alpha_h = 1.4  # tortuosity
        c_0 = 340.0

    if material=='BIRCHLT':
        phi = 0.529  # porosity
        gamma_p = 7.0 / 5.0
        sigma = 151429.0  # resitivity
        rho_0 = 1.2
        alpha_h = 1.37  # tortuosity
        c_0 = 340.0
 
 
    # isorel
    if material=='ISOREL':
        phi = 0.7  # porosity
        gamma_p = 7.0 / 5.0
        sigma = 9067.0  # resitivity
        rho_0 = 1.2
        alpha_h = 1.0  # tortuosity
        c_0 = 340.0

    #polyester
    if material=='POLYESTER':
        phi = 0.93  # porosity
        gamma_p = 7.0 / 5.0
        sigma = 124000.0  # resitivity
        rho_0 = 1.2
        alpha_h = 1.0  # tortuosity
        c_0 = 340.0

    # parameters of the geometry
    L = 2.5

    # parameters of the mesh
    resolution = 12  # := number of elements along L

    # parameters of the material (cont.)
    mu_0 = 1.0
    ksi_0 = 1.0 / (c_0 ** 2)
    mu_1 = phi / alpha_h
    ksi_1 = phi * gamma_p / (c_0 ** 2)
    a = sigma * (phi ** 2) * gamma_p / ((c_0 ** 2) * rho_0 * alpha_h)

    ksi_volume = phi * gamma_p / (c_0 ** 2)
    a_volume = sigma * (phi ** 2) * gamma_p / ((c_0 ** 2) * rho_0 * alpha_h)
    mu_volume = phi / alpha_h
    k2_volume = (1.0 / mu_volume) * ((omega ** 2) / (c_0 ** 2)) * (ksi_volume + 1j * a_volume / omega)
    print(k2_volume)

    # parameters of the objective function
    A = 1.0
    B = 1.0

    # defining k, omega and alpha dependant parameters' functions
    @Memoize
    def lambda_0(k, omega):
        if k ** 2 >= (omega ** 2) * ksi_0 / mu_0:
            return numpy.sqrt(k ** 2 - (omega ** 2) * ksi_0 / mu_0)
        else:
            return numpy.sqrt((omega ** 2) * ksi_0 / mu_0 - k ** 2) * 1j

    @Memoize
    def lambda_1(k, omega):
        temp1 = (omega ** 2) * ksi_1 / mu_1
        temp2 = numpy.sqrt((k ** 2 - temp1) ** 2 + (a * omega / mu_1) ** 2)
        real = (1.0 / numpy.sqrt(2.0)) * numpy.sqrt(k ** 2 - temp1 + temp2)
        im = (-1.0 / numpy.sqrt(2.0)) * numpy.sqrt(temp1 - k ** 2 + temp2)
        return complex(real, im)

        
    @Memoize
    def g(y):
        sigma=1.0
        y0=0.0
        return 10*numpy.exp(-(y-y0)**2/2*sigma**2)/(sigma*numpy.sqrt(2*numpy.pi))
    
    @Memoize
    def g_k(k):
        return quad_vec(lambda y: g(y) * numpy.exp(-1j * k * y), -6, 6)[0]/ (2*6)

    @Memoize
    def f(x, k):
        return ((lambda_0(k, omega) * mu_0 - x) * numpy.exp(-lambda_0(k, omega) * L) \
                + (lambda_0(k, omega) * mu_0 + x) * numpy.exp(lambda_0(k, omega) * L))

    @Memoize
    def chi(k, alpha, omega):
        return (g_k(k) * ((lambda_0(k, omega) * mu_0 - lambda_1(k, omega) * mu_1) \
                          / f(lambda_1(k, omega) * mu_1, k) - (lambda_0(k, omega) * mu_0 - alpha) / f(alpha, k)))

    @Memoize
    def eta(k, alpha, omega):
        return (g_k(k) * ((lambda_0(k, omega) * mu_0 + lambda_1(k, omega) * mu_1) \
                          / f(lambda_1(k, omega) * mu_1, k) - (lambda_0(k, omega) * mu_0 + alpha) / f(alpha, k)))

    @Memoize
    def e_k(k, alpha, omega):
        expm = numpy.exp(-2.0 * lambda_0(k, omega) * L)
        expp = numpy.exp(+2.0 * lambda_0(k, omega) * L)

        if k ** 2 >= (omega ** 2) * ksi_0 / mu_0:
            return ((A + B * (numpy.abs(k) ** 2)) \
                    * ( \
                                (1.0 / (2.0 * lambda_0(k, omega))) \
                                * ((numpy.abs(chi(k, alpha, omega)) ** 2) * (1.0 - expm) \
                                   + (numpy.abs(eta(k, alpha, omega)) ** 2) * (expp - 1.0)) \
                                + 2 * L * numpy.real(chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega)))) \
                    + B * numpy.abs(lambda_0(k, omega)) / 2.0 * ((numpy.abs(chi(k, alpha, omega)) ** 2) * (1.0 - expm) \
                                                                 + (numpy.abs(eta(k, alpha, omega)) ** 2) * (
                                                                             expp - 1.0)) \
                    - 2 * B * (lambda_0(k, omega) ** 2) * L * numpy.real(
                        chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega))))
        else:
            return ((A + B * (numpy.abs(k) ** 2)) * (L \
                                                     * ((numpy.abs(chi(k, alpha, omega)) ** 2) + (
                                numpy.abs(eta(k, alpha, omega)) ** 2)) \
                                                     + complex(0.0, 1.0) * (1.0 / lambda_0(k, omega)) * numpy.imag(
                        chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega) \
                                                          * (1.0 - expm))))) + B * L * (
                               numpy.abs(lambda_0(k, omega)) ** 2) \
                   * ((numpy.abs(chi(k, alpha, omega)) ** 2) + (numpy.abs(eta(k, alpha, omega)) ** 2)) \
                   + complex(0.0, 1.0) * B * lambda_0(k, omega) * numpy.imag(
                chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega) \
                                                  * (1.0 - expm)))

    @Memoize
    def sum_e_k(omega):
        def sum_func(alpha):
            s = 0.0
            for n in range(-resolution, resolution + 1):
                k = n * numpy.pi / L
                s += e_k(k, alpha, omega)
            return s

        return sum_func

    @Memoize
    def alpha(omega):
        alpha_0 = numpy.array(complex(40.0, -40.0))
        temp = real_to_complex(minimize(lambda z: numpy.real(sum_e_k(omega)(real_to_complex(z))), complex_to_real(alpha_0), tol=1e-4).x)
        print(temp, "------", "je suis temp")
        return temp

    @Memoize
    def error(alpha, omega):
        temp = numpy.real(sum_e_k(omega)(alpha))
        return temp

    temp_alpha = alpha(omega)
    temp_error = error(temp_alpha, omega)

    return temp_alpha, temp_error


def run_compute_alpha(material):
    print('Computing alpha...')
    numb_omega = 100  # 1000
    # omegas = numpy.logspace(numpy.log10(600), numpy.log10(30000), num=numb_omega)
    omegas = numpy.linspace(500, 4000, num=numb_omega)
    temp = [compute_alpha(omega, material=material) for omega in omegas]
    #print("temp:", "------", temp)
    alphas, errors = map(list, zip(*temp))
    alphas = numpy.array(alphas)
    errors = numpy.array(errors)

    print('Writing alpha...')
    output_filename = 'dta_omega_' + str(material) + '.mtx'
    scipy.io.mmwrite(output_filename, omegas.reshape(alphas.shape[0], 1), field='complex', symmetry='general')
    output_filename = 'dta_alpha_' + str(material) + '.mtx'
    scipy.io.mmwrite(output_filename, alphas.reshape(alphas.shape[0], 1), field='complex', symmetry='general')
    output_filename = 'dta_error_' + str(material) + '.mtx'
    scipy.io.mmwrite(output_filename, errors.reshape(errors.shape[0], 1), field='complex', symmetry='general')

    return


def run_plot_alpha(material):
    color = 'darkblue'

    print('Reading alpha...')
    input_filename = 'dta_omega_' + str(material) + '.mtx'
    omegas = scipy.io.mmread(input_filename)
    omegas = omegas.reshape(omegas.shape[0])
    input_filename = 'dta_alpha_' + str(material) + '.mtx'
    alphas = scipy.io.mmread(input_filename)
    alphas = alphas.reshape(alphas.shape[0])
    input_filename = 'dta_error_' + str(material) + '.mtx'
    errors = scipy.io.mmread(input_filename)
    errors = errors.reshape(errors.shape[0])

    print('Plotting alpha...')
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.subplot(1, 1, 1)
    matplotlib.pyplot.plot(numpy.real(omegas), numpy.real(alphas), color=color)
    matplotlib.pyplot.xlabel(r'$\omega$')
    matplotlib.pyplot.ylabel(r'$\operatorname{Re}(\alpha)$')
    matplotlib.pyplot.ylim(0, 35)
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('fig_alpha_real_' + str(material) + '.jpg')
    matplotlib.pyplot.close(fig)

    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.subplot(1, 1, 1)
    matplotlib.pyplot.plot(numpy.real(omegas), numpy.imag(alphas), color=color)
    matplotlib.pyplot.xlabel(r'$\omega$')
    matplotlib.pyplot.ylabel(r'$\operatorname{Im}(\alpha)$')
    matplotlib.pyplot.ylim(-120, 10)
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('fig_alpha_imag_' + str(material) + '.jpg')
    matplotlib.pyplot.close(fig)

    fig = matplotlib.pyplot.figure()
    ax = matplotlib.pyplot.axes()
    ax.fill_between(numpy.real(omegas), numpy.real(errors), color=color)
    matplotlib.pyplot.ylim(1.e-9, 1.e-4)
    matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.xlabel(r'$\omega$')
    matplotlib.pyplot.ylabel(r'$e(\alpha)$')
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('fig_error_' + str(material) + '.jpg')
    matplotlib.pyplot.close(fig)

    return


def run():
    liste_materials = ['MELAMINE FOAM', 'GLASS WOOL', 'Polyurethane foam', 'PINELT', 'SAPELILT', 'BIRCHLT', 'ISOREL', 'POLYESTER']
    #liste_materials = ['MELAMINE FOAM']
    for material in liste_materials:
        run_compute_alpha(material)
        run_plot_alpha(material)

    #plot real part of alpha for all materials on the same graph
    color_list = ['darkblue', 'darkgreen', 'blue', 'yellow', 'orange', 'pink', 'red', 'purple']
    print('Plotting all alphas together...')
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.subplot(1, 1, 1)
    for i, material in enumerate(liste_materials):
        input_filename = 'dta_omega_' + str(material) + '.mtx'
        omegas = scipy.io.mmread(input_filename)
        omegas = omegas.reshape(omegas.shape[0])
        input_filename = 'dta_alpha_' + str(material) + '.mtx'
        alphas = scipy.io.mmread(input_filename)
        alphas = alphas.reshape(alphas.shape[0])
        matplotlib.pyplot.plot(numpy.real(omegas), numpy.real(alphas), color=color_list[i], label=material)
    matplotlib.pyplot.xlabel(r'$\omega$')
    matplotlib.pyplot.ylabel(r'$\operatorname{Re}(\alpha)$')
    matplotlib.pyplot.ylim(0, 40)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('fig_all_alpha_real.jpg')
    matplotlib.pyplot.close(fig)

    #plot imaginary part of alpha for all materials on the same graph
    print('Plotting all alphas together...')
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.subplot(1, 1, 1)
    for i, material in enumerate(liste_materials):
        input_filename = 'dta_omega_' + str(material) + '.mtx'
        omegas = scipy.io.mmread(input_filename)
        omegas = omegas.reshape(omegas.shape[0])
        input_filename = 'dta_alpha_' + str(material) + '.mtx'
        alphas = scipy.io.mmread(input_filename)
        alphas = alphas.reshape(alphas.shape[0])
        matplotlib.pyplot.plot(numpy.real(omegas), numpy.imag(alphas), color=color_list[i], label=material)
    matplotlib.pyplot.xlabel(r'$\omega$')
    matplotlib.pyplot.ylabel(r'$\operatorname{Im}(\alpha)$')
    matplotlib.pyplot.ylim(-120, 10)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
    matplotlib.pyplot.savefig('fig_all_alpha_imag.jpg')
    matplotlib.pyplot.close(fig)

    return

if __name__ == '__main__':
    run()
    print('End.')
