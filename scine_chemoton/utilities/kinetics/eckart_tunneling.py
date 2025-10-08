#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import math

import scine_utilities as utils


class EckartTunneling:
    def __init__(self, imaginary_wavenumber: float, lhs_barrier: float, rhs_barrier: float):
        """
        This class calculates the Eckart tunneling function for an asymmetric Eckart model potential.
        See https://pubs.acs.org/doi/epdf/10.1021/j100809a040 for details.

        Parameters
        ----------
        imaginary_wavenumber : float
            The wavenumber of the transition state in cm^-1.
        lhs_barrier : float
            The LHS reaction barrier in Hartree.
        rhs_barrier : float
            The RHS reaction barrier in Hartree
        """
        no_swap = lhs_barrier < rhs_barrier
        self.__delta_V_1: float = max(0.0, lhs_barrier if no_swap else rhs_barrier)
        self.__delta_V_2: float = max(0.0, rhs_barrier if no_swap else lhs_barrier)
        self.__h_nu = abs(imaginary_wavenumber) * utils.HARTREE_PER_INVERSE_CENTIMETER
        self.__alpha_1 = 2.0 * math.pi * self.__delta_V_1 / self.__h_nu
        self.__alpha_2 = 2.0 * math.pi * self.__delta_V_2 / self.__h_nu
        tmp = self.__alpha_1 * self.__alpha_2 - 4 * math.pi * math.pi / 16
        self.__2pi_d = 2 * math.sqrt(abs(tmp))

    def calculate_tunneling_function(self, energy: float) -> float:
        """
        Calculate the transmission probability as a function of energy.

        Parameters
        ----------
        energy : float
            The energy in a.u.

        Returns
        -------
        The transmission probability.
        """
        zeta = energy / self.__delta_V_1
        factor = 1/math.sqrt(self.__alpha_1) + 1/math.sqrt(self.__alpha_2)
        two_pi_a = 2 * math.sqrt(self.__alpha_1 * zeta) / factor
        two_pi_b = 2 * math.sqrt(abs((zeta - 1) * self.__alpha_1 + self.__alpha_2)) / factor
        arg_1 = two_pi_a - two_pi_b
        arg_2 = two_pi_a + two_pi_b
        # Avoid calculating the cosh with a very large argument.
        if arg_1 > 200 or arg_2 > 200 or self.__2pi_d > 200:
            return self.kappa_lower_exponential(arg_1, arg_2)
        cosh_2pi_d = math.cosh(self.__2pi_d)
        return 1 - (math.cosh(arg_1) + cosh_2pi_d) / (math.cosh(arg_2) + cosh_2pi_d)

    def calculate_tunneling_function_alternative(self, energy: float) -> float:
        """
        Calculate the transmission probability as a function of energy. This function explicitly avoids
        large arguments in the cosh function of the transmission probability.

        Parameters
        ----------
        energy : float
            The energy in a.u.

        Returns
        -------
        The transmission probability.
        """
        zeta = energy / self.__delta_V_1
        factor = 1/math.sqrt(self.__alpha_1) + 1/math.sqrt(self.__alpha_2)
        two_pi_a = 2 * math.sqrt(self.__alpha_1 * zeta) / factor
        two_pi_b = 2 * math.sqrt(abs((zeta - 1) * self.__alpha_1 + self.__alpha_2)) / factor
        arg_1 = two_pi_a - two_pi_b
        arg_2 = two_pi_a + two_pi_b
        return self.kappa_lower_exponential(arg_1, arg_2)

    def kappa_lower_exponential(self, arg1: float, arg2: float) -> float:
        """
        Avoid cosh, expand it as an exponential, and divide every term by exp(max_arg). This should ensure
        that the argument for the exponential never becomes large. Doing this is a bit ugly in the code but
        still exact.

        Parameters
        ----------
        arg1 : float
            The first argument for the cosh.
        arg2 : float
            The second argument for the cosh.

        Returns
        -------
        The transmission probability.
        """
        max_arg = max(arg1, arg2, self.__2pi_d)
        tmp = math.exp(self.__2pi_d - max_arg) + math.exp(-self.__2pi_d - max_arg)
        nominator = math.exp(arg1 - max_arg) + math.exp(-arg1 - max_arg) + tmp
        denominator = math.exp(arg2 - max_arg) + math.exp(-arg2 - max_arg) + tmp
        return 1 - nominator / denominator
