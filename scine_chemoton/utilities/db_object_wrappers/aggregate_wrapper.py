#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Union, Optional, List, Tuple, Dict

import scine_database as db

from .ensemble_wrapper import Ensemble
from .thermodynamic_properties import ReferenceState
from scine_database.concentration_query_functions import query_concentration_with_object
from ..get_molecular_formula import get_elements_in_structure


class Aggregate(Ensemble):
    """
    A class that wraps around db.Compound and db.Flask to provide easy access to all thermodynamic contributions.
    Free energies are calculated with the harmonic oscillator, rigid rotor, particle in a box model according to the
    given thermodynamic reference state (allows caching or works on the fly if necessary).

    Parameters
    ----------
    aggregate_id : db.ID
        The database ID.
    manager
        The database manager.
    electronic_model
        The electronic structure model from which the electronic energy is taken.
    hessian_model
        The electronic structure model with which the geometrical hessian was calculated.
    only_electronic
        If true, only the electronic energies are used to determine the thermodynamics.
    """

    def __init__(self, aggregate_id: db.ID, manager: db.Manager, electronic_model: db.Model, hessian_model: db.Model,
                 only_electronic: bool = False) -> None:
        self._is_flask = False
        super().__init__(aggregate_id, manager, electronic_model, hessian_model, only_electronic)
        self._start_concentration_property_name = "start_concentration"
        self._concentration_flux_property_name = "concentration_flux"

    def _initialize_db_object(self, manager: db.Manager) -> Union[db.Compound, db.Flask]:
        compounds = manager.get_collection("compounds")
        db_object = db.Compound(self._db_id, compounds)  # type: ignore
        if not db_object.exists():
            flasks = manager.get_collection("flasks")
            db_object = db.Flask(self._db_id, flasks)  # type: ignore
            self._is_flask = True
        return db_object

    def _update_thermodynamics(self):
        """
        Updates thermodynamics container for every structure that should be analyzed.
        """
        structure_ids = self.get_db_object().get_structures()
        for s_id in structure_ids:
            structure = db.Structure(s_id, self._structures)
            if not structure.analyze():
                continue
            _ = self._structure_thermodynamics.get_or_produce(s_id)

    def get_free_energy(self, reference_state: ReferenceState) -> Optional[float]:
        """
        Getter for the aggregate's free energy.
        """
        self._update_thermodynamics()
        return self._structure_thermodynamics.get_ensemble_gibbs_free_energy(reference_state)

    def get_enthalpy(self, reference_state: ReferenceState) -> Optional[float]:
        """
        Getter for the aggregate's enthalpy.
        """
        self.get_free_energy(reference_state)  # update if necessary
        return self._structure_thermodynamics.get_ensemble_enthalpy(reference_state)

    def get_entropy(self, reference_state: ReferenceState) -> Optional[float]:
        """
        Getter for the aggregate's entropy.
        """
        self.get_free_energy(reference_state)  # update if necessary
        return self._structure_thermodynamics.get_ensemble_entropy(reference_state)

    def get_starting_concentration(self) -> float:
        """
        Getter for the aggregates starting concentration.
        """
        return query_concentration_with_object(self._start_concentration_property_name, self.get_db_object(),
                                               self._properties, self._structures)

    def get_concentration_flux(self) -> float:
        """
        Getter for the aggregates vertex flux.
        """
        return query_concentration_with_object(self._concentration_flux_property_name, self.get_db_object(),
                                               self._properties, self._structures)

    def get_aggregate_type(self) -> db.CompoundOrFlask:
        """
        Getter for the aggregate type.
        """
        return db.CompoundOrFlask.FLASK if self._is_flask else db.CompoundOrFlask.COMPOUND

    def get_sorted_structure_list(self, reference_state: ReferenceState) -> List[Tuple[db.ID, float]]:
        """
        Getter a list of structures and their energies sorted by energy (ascending).

        Parameters
        ----------
        reference_state : ReferenceState
            The thermodynamic reference state for the free energy approximation.
        """
        self.get_free_energy(reference_state)  # update if necessary
        return self._structure_thermodynamics.get_sorted_structure_list(reference_state)

    def get_element_count(self) -> Dict[str, int]:
        """
        Getter for a dictionary with the element and its count in the aggregate.
        """
        structure = db.Structure(self.get_db_object().get_centroid(), self._structures)
        return get_elements_in_structure(structure)
