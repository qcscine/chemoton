#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Union, Dict, List, Tuple
from abc import ABC, abstractmethod

import scine_database as db
import scine_utilities as utils

from .thermodynamic_properties import ThermodynamicPropertiesCache, ReferenceState
from ..model_combinations import ModelCombination


class Ensemble(ABC):
    """
    Base class of the wrapper for an ensemble of structures that is derived from either a database compound, flask, or
    reaction.

    Parameters
    ----------
    db_id : db.ID
        The database ID.
    manager : db.Manager
        The database manager.
    electronic_model : db.Model
        The electronic structure model for the electronic energy contribution to the free energy.
    hessian_model : db.Model
        The electronic structure model with which the vibrational (Hessian) and other free energy corrections are
        calculated.
    only_electronic : bool
        If true, only the electronic energy is considered for the free energies.
    """

    def __init__(self, db_id: db.ID, manager: db.Manager, electronic_model: db.Model, hessian_model: db.Model,
                 only_electronic: bool = False) -> None:
        self._electronic_model = electronic_model
        self._hessian_model = hessian_model
        self._only_electronic = only_electronic
        self._structures = manager.get_collection("structures")
        self._properties = manager.get_collection("properties")
        self._db_id = db_id
        self._db_object = self._initialize_db_object(manager)
        self._manager = manager
        assert self._db_object.exists()
        self._structure_thermodynamics = ThermodynamicPropertiesCache(self._structures, self._properties,
                                                                      self._electronic_model, self._hessian_model,
                                                                      self._only_electronic)

    @abstractmethod
    def _initialize_db_object(self, manager: db.Manager) -> Union[db.Compound, db.Flask, db.Reaction]:
        raise NotImplementedError

    def get_db_id(self) -> db.ID:
        """
        Returns the database ID.
        """
        return self._db_id

    def get_manager(self) -> db.Manager:
        """
        Returns the database manager.
        """
        return self._manager

    def __int__(self):
        """
        Return the database ID as an integer.
        """
        return int(self.get_db_id().string(), 16)

    def __eq__(self, other) -> bool:
        """
        Equal operator. Comparse the database IDs.
        """
        return self._db_id == other.get_db_id()

    def __hash__(self):
        """
        Return a hash of the object based on its ID.
        """
        return hash(self.__int__())

    def get_db_object(self):
        """
        Return the database object.
        """
        assert self._db_object
        return self._db_object

    @abstractmethod
    def _update_thermodynamics(self):
        raise NotImplementedError

    def complete(self):
        """
        Returns true if a free energy approximation is available for the ensemble. Returns false otherwise.
        """
        self._update_thermodynamics()
        return self._structure_thermodynamics.provides_values()

    def get_electronic_model(self):
        """
        Getter for the electronic structure mode with which the electronic energies are evaluated.
        """
        return self._electronic_model

    def analyze(self):
        """
        Returns true if the database object is set to analyze. Returns false otherwise.
        """
        return self.get_db_object().analyze()

    def explore(self):
        """
        Returns true if the database object is set to explore. Returns false otherwise.
        """
        return self.get_db_object().explore()

    def get_model_combination(self):
        """
        Getter for the model combination of electronic and Hessian model.
        """
        return ModelCombination(self._electronic_model, self._hessian_model)

    @abstractmethod
    def get_element_count(self) -> Dict[str, int]:
        """
        Getter for a dictionary with the element and its count in the aggregate.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sorted_structure_list(self, _: ReferenceState) -> List[Tuple[db.ID, float]]:
        """
        Getter a list of structures and their energies sorted by energy (ascending).

        Parameters
        ----------
        _ : ReferenceState
            The thermodynamic reference state for the free energy approximation.
        """
        raise NotImplementedError

    def get_lowest_n_structures(self, n: int, energy_cut_off: float, reference_state: ReferenceState) -> List[db.ID]:
        """
        Getter for the lowest n structures according to their energy.

        Parameters
        ----------
        n : int
            The number of structures requested.
        energy_cut_off : float
            The energy cutoff after which structures are no longer considered for the list.
        reference_state : ReferenceState
            The thermodynamic reference state for the free energy approximation.

        Returns
        -------
        List[Tuple[db.ID, float]]
            The lowest n structures according to their free energy approximation.
        """
        s_id_and_gibbs = self.get_sorted_structure_list(reference_state)
        min_gibbs = s_id_and_gibbs[0][1] if s_id_and_gibbs else 0.0
        threshold = energy_cut_off * utils.KJPERMOL_PER_HARTREE
        structure_ids: List[db.ID] = []
        for s_id, gibbs in s_id_and_gibbs:
            if gibbs - min_gibbs <= threshold and len(structure_ids) < n:
                structure_ids.append(s_id)
        return structure_ids
