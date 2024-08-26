#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Optional, Dict, List, Tuple
import math

import scine_database as db
import scine_utilities as utils


class ReferenceState:
    """
    Reference state for thermodynamic calculations (harmonic vib., rigid rotor, particle in a box).
    Requires pressure, temperature, and optionally the symmetry of the molecule.

    Parameters
    ----------
    t : float
        The temperature in K.
    p : float
        The pressure in Pa.
    sym : int
        The symmetry number of the molecule.
    """

    def __init__(self, t: float, p: float, sym: int = 1) -> None:
        self.temperature = t
        self.pressure = p
        self.symmetry = sym
        assert isinstance(self.temperature, float)
        assert isinstance(self.pressure, float)
        assert isinstance(self.symmetry, int)

    def __eq__(self, other):
        """
        Equal operator. Compares temperature, pressure, and symmetry number.
        """
        if not isinstance(other, ReferenceState):
            return False
        return abs(self.temperature - other.temperature) < 1e-9 and abs(self.pressure - other.pressure) < 1e-9\
            and self.symmetry - other.symmetry == 0


class PlaceHolderReferenceState(ReferenceState):
    """
    Place-holder reference state. This can be used as a replacement for None default arguments that must be replaced
    at a later point.
    """

    def __init__(self) -> None:
        super().__init__(t=math.nan, p=math.nan, sym=1)

    def __eq__(self, other):
        return isinstance(other, PlaceHolderReferenceState)


class ThermodynamicProperties:
    """
    A class that provides easy access to free energy and electronic energy contributions of a structure.

    Parameters
    ----------
    hessian_property_id
        The ID of the hessian property.
    energy_property_id
        The ID of the electronic energy property.
    property_collection
        The property collection.
    structure_collection
        The structure collection.
    """

    def __init__(self, hessian_property_id: Optional[db.ID], energy_property_id: db.ID,
                 property_collection: db.Collection, structure_collection: db.Collection,
                 structure_id: db.ID) -> None:
        self._hessian_prop_id = hessian_property_id
        self._energy_property_id = energy_property_id
        self._structure_id = structure_id
        self._properties = property_collection
        self._structures = structure_collection
        self._electronic_energy: Optional[float] = None
        self._entropy: Optional[float] = None
        self._enthalpy: Optional[float] = None
        self._zero_point_energy_correction: Optional[float] = None
        self._reference_state = ReferenceState(float("inf"), float("inf"), 1)

    def get_thermochemistry_calculator(self) -> utils.ThermochemistryCalculator:
        """
        Getter for the ThermochemistryCalculator.
        """
        assert self._hessian_prop_id
        self._electronic_energy = self.get_electronic_energy()
        assert self._electronic_energy is not None
        hessian_prop = db.DenseMatrixProperty(self._hessian_prop_id, self._properties)
        hessian = hessian_prop.get_data()
        structure = db.Structure(hessian_prop.get_structure(), self._structures)
        atom_collection = structure.get_atoms()
        return utils.ThermochemistryCalculator(hessian, atom_collection, structure.get_multiplicity(),
                                               self._electronic_energy)

    def _update_thermodynamics(self, reference_state: ReferenceState):
        if reference_state == self._reference_state:
            return
        self._reference_state = reference_state
        thermochemistry_calculator = self.get_thermochemistry_calculator()
        thermochemistry_calculator.set_temperature(self._reference_state.temperature)
        thermochemistry_calculator.set_pressure(self._reference_state.pressure)
        thermochemistry_calculator.set_molecular_symmetry(self._reference_state.symmetry)
        thermochemistry_results = thermochemistry_calculator.calculate()
        self._entropy = thermochemistry_results.overall.entropy
        self._enthalpy = thermochemistry_results.overall.enthalpy
        self._zero_point_energy_correction = thermochemistry_results.overall.zero_point_vibrational_energy
        assert self._enthalpy
        assert self._entropy

    def get_reference_state(self) -> ReferenceState:
        """
        Getter for the last thermodynamic reference state.
        """
        return self._reference_state

    def get_electronic_energy(self) -> float:
        """
        Getter for the electronic energy.
        """
        if self._electronic_energy is None:
            self._electronic_energy = db.NumberProperty(self._energy_property_id, self._properties).get_data()
        return self._electronic_energy

    def get_enthalpy(self, reference_state: ReferenceState) -> float:
        """
        Getter for the enthalpy (may update the reference state).

        Parameters
        ----------
        reference_state
            The thermodynamic reference state (temperature/pressure).
        """
        self._update_thermodynamics(reference_state)
        assert self._enthalpy
        return self._enthalpy

    def get_entropy(self, reference_state: ReferenceState) -> float:
        """
        Getter for the entropy (may update the reference state).

        Parameters
        ----------
        reference_state
            The thermodynamic reference state (temperature/pressure).
        """
        self._update_thermodynamics(reference_state)
        assert self._entropy
        return self._entropy

    def get_gibbs_free_energy_correction(self, reference_state: ReferenceState) -> float:
        """
        Getter for the gibbs free energy correction (may update the reference state).

        Parameters
        ----------
        reference_state
            The thermodynamic reference state (temperature/pressure).
        """
        return self.get_gibbs_free_energy(reference_state) - self.get_electronic_energy()

    def get_gibbs_free_energy(self, reference_state: ReferenceState) -> float:
        """
        Getter for the gibbs free energy (may update the reference state).

        Parameters
        ----------
        reference_state
            The thermodynamic reference state (temperature/pressure).
        """
        h = self.get_enthalpy(reference_state)
        s = self.get_entropy(reference_state)
        # If the temperature is 0 K, the entropy diverges. To avoid NaNs, we then only take the enthalpy.
        return h - reference_state.temperature * s if reference_state.temperature > 1e-6 else h

    def get_zero_point_energy(self) -> float:
        """
        Getter for the vibrational zero point energy.
        """
        self._update_thermodynamics(self._reference_state)
        return self.get_electronic_energy() + self.get_zero_point_energy_correction()

    def get_zero_point_energy_correction(self) -> float:
        """
        Getter for the vibrational zero point energy correction.
        """
        self._update_thermodynamics(self._reference_state)
        assert self._zero_point_energy_correction
        return self._zero_point_energy_correction

    def id(self) -> db.ID:
        """
        Getter for the structure's database ID.
        """
        return self._structure_id


class ThermodynamicPropertiesCache:
    """
    The purpose of this class is to provide access to all ThermodynamicProperties of an aggregate or set of
    transition states which provide access to the thermodynamic properties of some structure. Furthermore, it
    provides access to the minimum Gibbs free energy, enthalpy, entropy etc. since it effectively represents a
    structure ensemble.

    Parameters
    ----------
    structures
        The structure collection.
    properties
        The property collection
    electronic_model
        The model for the electronic energies.
    hessian_model
        The model for the hessian correction. May be None.
    only_electronic
        If true, only the electronic energies are used to characterize the structures.

    TODO: Add conformational free energy contributions after flask deduplication.
    """

    def __init__(self, structures: db.Collection, properties: db.Collection,
                 electronic_model: db.Model, hessian_model: db.Model, only_electronic: bool = False) -> None:
        self._only_electronic = only_electronic
        self._structures = structures
        self._properties = properties
        self._electronic_model = electronic_model
        self._hessian_model = hessian_model
        self._cache: Dict[int, ThermodynamicProperties] = {}
        # Structure IDs and energy approximations. The list is sorted by the free energy (ascending).
        self._sorted_list: List[Tuple[db.ID, float]] = []
        self._electronic_energy_property_name = "electronic_energy"
        self._hessian_property_name = "hessian"

        self._reference_state: ReferenceState = ReferenceState(float("inf"), float("inf"))
        self._minimum_gibbs: Optional[float] = None
        self._minimum_enthalpy: Optional[float] = None
        self._minimum_entropy: Optional[float] = None
        self._n_structures = 0
        self._minimum_structure_int_id: int = 0

    def get_or_produce(self, structure_id: db.ID) -> Optional[ThermodynamicProperties]:
        """
        Getter for the thermodynamic properties of a structure.

        Parameters
        ----------
        structure_id
            The structure ID.

        Returns
        -------
            The corresponding ThermodynamicProperties object. May return None, if the structure lacks some free energy
            contribution.
        """
        int_id = int(structure_id.string(), 16)
        if int_id not in self._cache:
            structure = db.Structure(structure_id, self._structures)
            if structure.label == db.Label.DUPLICATE:
                return None
            energy_property_ids = structure.query_properties(self._electronic_energy_property_name,
                                                             self._electronic_model, self._properties)
            if not energy_property_ids or not db.NumberProperty(energy_property_ids[-1], self._properties).analyze():
                return None
            hessian_property_ids = structure.query_properties(self._hessian_property_name,
                                                              self._hessian_model, self._properties)
            if not self._only_electronic and not hessian_property_ids:
                return None
            hess_id = None if self._only_electronic else hessian_property_ids[-1]
            if hess_id and not db.DenseMatrixProperty(hess_id, self._properties).analyze():
                return None
            thermo = ThermodynamicProperties(hess_id, energy_property_ids[-1], self._properties, self._structures,
                                             structure_id)
            self._cache[int_id] = thermo
        return self._cache[int_id]

    def minimum_values_need_update(self, reference_state, n_structures: Optional[int] = None) -> bool:
        """
        Check if the cached minimum values must be updated.

        Parameters
        ----------
        reference_state
            The reference state (temperature/pressure).
        n_structures
            (optional) The number of elements the cache should contain.

        Returns
        -------
            Returns true, if the cache is out of date or incomplete.
        """
        if n_structures is None:
            n_structures = self._n_structures
        return reference_state != self._reference_state or n_structures != self._n_structures

    def _update_minimum(self, reference_state: ReferenceState) -> None:
        if not self.minimum_values_need_update(reference_state, len(self._cache.keys())):
            return
        self._sorted_list = []
        for therm in self._cache.values():
            energy = therm.get_electronic_energy() if self._only_electronic else therm.get_gibbs_free_energy(
                reference_state)
            self._sorted_list.append((therm.id(), energy))
        self._sorted_list.sort(key=lambda x: x[1])
        if self._sorted_list:
            self._minimum_structure_int_id = int(self._sorted_list[0][0].string(), 16)
            self._minimum_gibbs = self._sorted_list[0][1]
            self._n_structures = len(self._cache.keys())
            self._reference_state = reference_state
            therm = self._cache[self._minimum_structure_int_id]
            if not self._only_electronic:
                self._minimum_enthalpy = therm.get_enthalpy(reference_state)
                self._minimum_entropy = therm.get_entropy(reference_state)
            else:
                self._minimum_enthalpy = self._minimum_gibbs
                self._minimum_entropy = 0.0

    def get_n_cached(self) -> int:
        """
        Getter for the number of cached elements.
        """
        return len(self._cache.keys())

    def get_ensemble_gibbs_free_energy(self, reference_state) -> Optional[float]:
        """
        Getter for the Gibb's free energy of the underlying structure ensemble.

        Returns
        -------
            At the moment only the minimum value is returned. May return None, if no free energy is available for any
            structure.
        """
        self._update_minimum(reference_state)
        return self._minimum_gibbs

    def get_ensemble_enthalpy(self, reference_state: ReferenceState) -> Optional[float]:
        """
        Getter for the enthalpy of the structure with the lowest Gibb's free energy approximation (no conformational
        contribution).

        Returns
        -------
            The enthalpy of the structure with the lowest Gibb's free energy approximation. May return None, if no
            enthalpy is available.
        """
        self.get_ensemble_gibbs_free_energy(reference_state)
        return self._minimum_enthalpy

    def get_ensemble_entropy(self, reference_state) -> Optional[float]:
        """
        Getter for the entropy of the structure with the lowest Gibb's free energy approximation (no conformational
        contribution).

        Returns
        -------
            The entropy of the structure with the lowest Gibb's free energy approximation. May return None, if no
            entropy is available.
        """
        self.get_ensemble_gibbs_free_energy(reference_state)
        return self._minimum_entropy

    def get_sorted_structure_list(self, reference_state: ReferenceState) -> List[Tuple[db.ID, float]]:
        self.get_ensemble_gibbs_free_energy(reference_state)
        return self._sorted_list

    def clear(self):
        """
        Clear and reset cache.
        """
        self._cache = {}
        self._reference_state = ReferenceState(float("inf"), float("inf"))
        self._minimum_gibbs = None
        self._minimum_enthalpy = None
        self._minimum_entropy = None
        self._n_structures = 0
        self._minimum_structure_int_id = 0
        self._sorted_list = []

    def provides_values(self) -> bool:
        """
        Returns True if values other than None can be provided by this cache.
        """
        return len(self._cache.keys()) != 0
