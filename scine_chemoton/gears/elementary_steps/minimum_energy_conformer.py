#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import Tuple, List
import math

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application imports
from . import ElementaryStepGear
from ...utilities.energy_query_functions import get_energy_for_structure


class MinimumEnergyConformerElementarySteps(ElementaryStepGear):
    """
    This Gear probes Reactions by trying to react the minimum energy conformer of each
    Compound with the minimum energy conformer of the other Compound.
    For each Structure--Structure combination multiple arrangements (possible
    Elementary Steps) will be tested.

    This Gear does not consider Flasks/Complexes as reactive, they are not probed
    for elementary steps.

    Attributes
    ----------
    options :: Options
        The options for the gear.
    aggregate_filter :: scine_chemoton.gears.elementary_steps.aggregate_filters.AggregateFilter
        A filter for allowed reaction combinations, per default everything
        is permitted, no filter is applied.
    trial_generator :: TrialGenerator
        The generator to set up elementary step trial calculations by enumerating
        reactive complexes and trial reaction coordinates

    Notes
    -----
    This function assumes maximum spin when adding two Structures into one
    reactive complex.
    The need for elementary step guesses is tested by:

    a. for bimolecular reactions: checking whether there is already a
        calculation to search for a bimolecular reaction of the same
        structures with the same job order
    b. for unimolecular reactions: checking whether there is already a
        calculation to search for an intramolecular reaction of the same
        structure  with the same job order
    """

    class Options(ElementaryStepGear.Options):

        __slots__ = ("energy_upper_bound", "max_number_structures")

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.energy_upper_bound = 12.0
            """
            float
                Upper bound for the energy difference to the conformer with the lowest energy to
                be considered in the reaction trial generation (in kJ/mol). Default is 12 kJ/mol
                which corresponds to an occupation change of ~ 1 % (according to their Boltzmann
                exponential factors).
            """
            self.max_number_structures = math.inf
            """
            int
                The maximum number of structures considered for each compound.
            """

    options: Options

    def __init__(self):
        super().__init__()
        self._energy_label = "electronic_energy"
        """
        str
            The property label for the energy property.
        """

    def _check_if_conformers_are_present(self, compound) -> bool:
        centroid = db.Structure(compound.get_centroid(), self._structures)
        return centroid.has_property("boltzmann_weight")

    def _get_all_energies_and_structure_ids(self, compound: db.Compound) -> List[Tuple[float, db.ID]]:
        energies_sids: List[Tuple[float, db.ID]] = list()
        for sid in compound.get_structures():
            structure = db.Structure(sid, self._structures)
            # Only consider optimized structures, no guess structures or duplicates
            if not structure.explore() or structure.get_label() not in [
                db.Label.MINIMUM_OPTIMIZED,
                db.Label.USER_OPTIMIZED,
            ]:
                continue
            energy = get_energy_for_structure(structure, self._energy_label, self.options.model,
                                              self._structures, self._properties)
            # There may not be an energy available for every structure at the given model.
            if energy is not None:
                energies_sids.append((energy, sid))
        return energies_sids

    def _get_eligible_structures(self, compound: db.Compound) -> List[db.ID]:
        if not self._check_if_conformers_are_present(compound):
            return []
        energies_s_ids = self._get_all_energies_and_structure_ids(compound)
        if not energies_s_ids:
            return []
        energies_s_ids_sorted = sorted(energies_s_ids, key=lambda tup: tup[0])
        minimum_energy = min(energies_s_ids_sorted, key=lambda tup: tup[0])[0]
        eligible_sids = list()
        n_added = 0
        threshold = self.options.energy_upper_bound * utils.HARTREE_PER_KJPERMOL
        for energy, s_id in energies_s_ids_sorted:
            if abs(energy - minimum_energy) <= threshold:
                eligible_sids.append(s_id)
                n_added += 1
            if n_added >= self.options.max_number_structures:
                break
        return eligible_sids
