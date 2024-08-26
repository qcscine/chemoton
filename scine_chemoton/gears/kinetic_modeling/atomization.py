#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Dict, Union, Any
import numpy as np
from copy import deepcopy

import scine_database as db
import scine_utilities as utils

from scine_database.energy_query_functions import get_energy_for_structure
from scine_chemoton.utilities.db_object_wrappers.aggregate_wrapper import Aggregate
from scine_chemoton.utilities.calculation_creation_helpers import finalize_calculation
from scine_chemoton.utilities.get_molecular_formula import get_elements_in_structure
from scine_database.queries import calculation_exists_in_structure, stop_on_timeout


class ZeroEnergyReference:
    """
    Default energy zero point of the electronic structure method. The method is always taken without any solvent/
    solvation settings.
    """

    def __init__(self, model) -> None:
        self._model_without_solvent = deepcopy(model)
        self._model_without_solvent.solvation = "None"
        self._model_without_solvent.solvent = ""

    def model_matches(self, model: db.Model):
        """
        Return true, if the electronic structure models match.
        """
        model_without_solvent = deepcopy(model)
        model_without_solvent.solvation = "None"
        model_without_solvent.solvent = ""
        return model_without_solvent == self._model_without_solvent

    def get_model(self):
        """
        Getter for the underlying model.
        """
        return self._model_without_solvent

    def get_reference_energy(self, _):
        """
        Getter for the electronic energy reference.
        """
        return 0.0


class AtomEnergyReference(ZeroEnergyReference):
    """
    Reference the energy to the single atom energies to calculate energies of formation. You can calculate the energy
    of formation as energy - reference.get_reference_energy(structure_id).
    """

    def __init__(self, model: db.Model, manager: db.Manager, atom_energies=None) -> None:
        super().__init__(model)
        if atom_energies is None:
            atom_energies = {}
        self._atom_energies = atom_energies
        self._electronic_energy_key = "electronic_energy"
        self._manager: db.Manager = manager
        self._structures = self._manager.get_collection("structures")
        self._compounds = self._manager.get_collection("compounds")
        self._properties = self._manager.get_collection("properties")
        self._atom_structures: Dict[str, db.Structure] = {}
        self.single_atom_multiplicities: Dict[str, int] = {}
        self._single_point_energy_job = db.Job("scine_single_point")
        self.settings: Dict[str, Any] = {}

    def get_reference_energy(self, structure_id):
        structure = db.Structure(structure_id, self._structures)
        counts = get_elements_in_structure(structure)
        atom_energies = 0.0
        for e, n in counts.items():
            atom_energies += n * self.get_atom_energy(e)
        return atom_energies

    def get_atom_energy(self, element: str):
        if element not in self._atom_energies:
            self._update_atom_structures()
            if element not in self._atom_energies:
                raise RuntimeError("Atom energy not available for element " + element + " and model "
                                   + str(self._model_without_solvent))
            structure = self._atom_structures[element]
            energy = get_energy_for_structure(structure, self._electronic_energy_key, self._model_without_solvent,
                                              self._structures, self._properties)
            if energy is None:
                raise RuntimeError("Atom energy not available for element " + element + " and model "
                                   + str(self._model_without_solvent))
            self._atom_energies[element] = energy
        return self._atom_energies[element]

    def _update_atom_structures(self):
        for compound in stop_on_timeout(self._compounds.iterate_all_compounds()):
            compound.link(self._compounds)
            centroid = db.Structure(compound.get_centroid(), self._structures)
            atom_collection = centroid.get_atoms()
            if len(atom_collection) != 1:
                continue
            centroid_element = str(atom_collection.elements[0])
            if centroid_element in self._atom_structures:
                continue
            self._atom_structures[centroid_element] = centroid

    def _get_atom_multiplicity(self, element: str):
        if element not in self.single_atom_multiplicities:
            raise LookupError("Error: Single atom multiplicity missing. Please provide all single atom multiplicities"
                              " manually.")
        return self.single_atom_multiplicities[element]

    def set_up_atom_energy_calculations(self, reference_structure_ids: List[db.ID]):
        """
        Set up all atom energy calculations for the elements present in the given structures if these calculations
        where not already set up.
        """
        self._update_atom_structures()
        structures = self._manager.get_collection("structures")
        calculations = self._manager.get_collection("calculations")
        elements: List[str] = []
        for s_id in reference_structure_ids:
            structure = db.Structure(s_id, structures)
            for e in structure.get_atoms().elements:
                e_str = str(e)
                if e_str not in elements:
                    elements.append(e_str)
        for element in elements:
            if element in self._atom_energies:
                continue
            if element not in self._atom_structures:
                coords = np.zeros((1, 3))
                atom_collection = utils.AtomCollection([utils.ElementInfo.element_from_symbol(element)], coords)
                atom = db.Structure.make(atom_collection, 0, self._get_atom_multiplicity(element), structures)
                self._atom_structures[element] = atom
            structure_ids = [self._atom_structures[element].id()]
            s = self.settings if self.settings else None
            if calculation_exists_in_structure(self._single_point_energy_job.order, structure_ids,
                                               self._model_without_solvent, structures, calculations, s):
                continue
            calculation = db.Calculation()
            calculation.link(calculations)
            calculation.create(self._model_without_solvent, self._single_point_energy_job, structure_ids)
            calculation.set_priority(1)
            if self.settings:
                calculation.set_settings(utils.ValueCollection(self.settings))
            finalize_calculation(calculation, structures, structure_ids)
            calculation.set_status(db.Status.NEW)


class MultiModelEnergyReferences:
    """
    Energy references for a combination of electronic structure models. The correct energy reference is determind
    on the fly for the given model. Energy references are only allowed once for every model.
    """

    def __init__(self, references: List[Union[ZeroEnergyReference, AtomEnergyReference]]) -> None:
        self._references = []
        for ref in references:
            if not self.has_reference(ref.get_model()):
                self._references.append(ref)

    def has_reference(self, model: db.Model):
        """
        Returns true, if there is a fitting model.
        """
        for ref in self._references:
            if ref.model_matches(model):
                return True
        return False

    def get_energy_reference(self, model: db.Model):
        """
        Getter for the electronic energy reference object with the given model.
        """
        for ref in self._references:
            if ref.model_matches(model):
                return ref
        raise RuntimeError("Error: No energy reference available for the given model: " + str(model))

    def get_value(self, aggregate: Aggregate) -> float:
        """
        Getter for the value of the electronic energy reference for the given aggregate wrapper.
        """
        centroid = aggregate.get_db_object().get_centroid()
        return self.get_energy_reference(aggregate.get_electronic_model()).get_reference_energy(centroid)


class PlaceHolderMultiModelEnergyReferences(MultiModelEnergyReferences):
    """
    A place-holder for a MultiModelEnergyReferences object that can be used instead of None.
    """

    def __init__(self) -> None:
        super().__init__([])

    def has_reference(self, _: db.Model):
        raise NotImplementedError

    def get_value(self, _: Aggregate) -> float:
        raise NotImplementedError

    def get_energy_reference(self, _: db.Model) -> float:
        raise NotImplementedError
