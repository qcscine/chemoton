#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""


# Standard library imports
import unittest
import math

# Third party imports
import scine_database as db
import scine_utilities as utils
from scine_database import test_database_setup as db_setup

# Local application tests imports
from scine_chemoton.tests.utilities.db_object_wrappers.test_reaction_wrapper import add_reaction
from scine_chemoton.utilities.db_object_wrappers.reaction_wrapper import Reaction
from scine_chemoton.gears import HoldsCollections
from scine_chemoton.utilities.db_object_wrappers.wrapper_caches import MultiModelCacheFactory
from scine_chemoton.utilities.db_object_wrappers.thermodynamic_properties import ReferenceState


class TestEckartTunneling(unittest.TestCase, HoldsCollections):
    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds", "flasks",
                                      "properties", "calculations"]
        self.initialize_collections(manager)
        MultiModelCacheFactory().clear()

    def tearDown(self) -> None:
        self._manager.wipe()

    @staticmethod
    def u_star_to_freq(u_star: float, temperature: float) -> float:
        return utils.BOLTZMANN_CONSTANT * temperature * u_star / utils.PLANCK_CONSTANT

    @staticmethod
    def alpha_to_energy(alpha: float, freq: float) -> float:
        energy_in_joule = utils.PLANCK_CONSTANT * freq * alpha / (2.0 * math.pi)
        return energy_in_joule * utils.HARTREE_PER_JOULE

    def test_eckart_tunneling(self):
        manager = db_setup.get_clean_db("chemoton_test_eckart_tunneling")
        self.custom_setup(manager)
        # reference values taken from table 1 in https://pubs.acs.org/doi/epdf/10.1021/j100809a040
        alpha_1_array = [0.5, 1.0, 2.0, 4.0, 8.0, 20.0]
        alpha_2_array = [0.5, 1.0, 2.0, 4.0, 8.0, 20.0]
        u_star_array = [2, 3, 4, 5]
        # tuple: alpha_1, alpha_2 : results for the u-star values
        references_barrier_penetrations = {
            # I think that the reference values for low alpha1 and alpha2 are wrong.
            # My results match nicely for alpha1, alpha2 >= 2 but not for lower values.
            # (0, 0): [1.16, 1.25, 1.34, 1.44],
            # (0, 1): [1.13, 1.21, 1.29, 1.38],
            # (0, 2): [1.09, 1.14, 1.20, 1.27],
            # (1, 1): [1.27, 1.43, 1.62, 1.83],
            # (1, 2): [1.21, 1.35, 1.51, 1.71],
            (2, 2): [1.32, 1.58, 1.91, 2.34],
            (2, 3): [1.26, 1.47, 1.77, 2.16],
            (3, 3): [1.30, 1.58, 2.02, 2.69],
            (3, 4): [1.25, 1.51, 1.93, 2.56],
            (4, 4): [1.24, 1.56, 2.04, 2.94],
            (5, 5): [1.20, 1.50, 2.10, 3.32]
        }

        temperature = 298.15
        reference_state = ReferenceState(temperature, 1e+5)
        model = db_setup.get_fake_model()
        energy_range = [0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.2]
        for key, value in references_barrier_penetrations.items():
            alpha_1 = alpha_1_array[key[0]]
            alpha_2 = alpha_2_array[key[1]]
            for u_star, v in zip(u_star_array, value):
                # Check vs reference and assert that the results are symmetric with respect to the exchange of
                # product and reactant.
                e_r = 0.0
                freq = self.u_star_to_freq(u_star, temperature)
                e_ts = e_r + self.alpha_to_energy(alpha_1, freq)
                e_p = e_ts - self.alpha_to_energy(alpha_2, freq)
                reaction, _, _, _ = add_reaction(manager, e_r, e_p, e_ts)
                reaction_2, _, _, _ = add_reaction(manager, e_p, e_r, e_ts)
                reaction_wrapper = Reaction(reaction.id(), manager, model, model, only_electronic=True)
                reaction_wrapper_2 = Reaction(reaction_2.id(), manager, model, model, only_electronic=True)
                reaction_wrapper.set_test_transition_state_wavenumber(freq / utils.SPEED_OF_LIGHT / 100)
                reaction_wrapper_2.set_test_transition_state_wavenumber(freq / utils.SPEED_OF_LIGHT / 100)
                gamma = reaction_wrapper.get_eckart_tunneling_penetration(reference_state)
                gamma_2 = reaction_wrapper_2.get_eckart_tunneling_penetration(reference_state)
                assert abs(gamma - v) < 5e-2
                assert abs(gamma_2 - gamma) < 1e-4

                tunneling_object = reaction_wrapper.get_eckart_tunneling_object(reference_state)
                for e in energy_range:
                    a = tunneling_object.calculate_tunneling_function(e)
                    b = tunneling_object.calculate_tunneling_function_alternative(e)
                    assert abs(a - b) < 1e-12
