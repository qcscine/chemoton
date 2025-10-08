#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from copy import deepcopy
from json import dumps
from typing import List, Optional, Tuple

# Third party imports
import numpy as np
import scine_database as db
from scine_database.queries import model_query

# Local application imports
from ....utilities.reactive_complexes.adsorption import AdsorptionGenerator
from ....utilities.reactive_complexes.adsorption import AdsorptionResult
from ....utilities.surfaces.pymatgen_interface import PmgInterface
from ....utilities.surfaces.periodic_utils import PeriodicUtils
from .connectivity_analyzer import ReactionType
from .fragment_based import FragmentBased
from . import TrialGenerator, _sanity_check_wrapper


class Adsorptions(FragmentBased):
    """
    Class to generate reactive complex calculations for adsorption reactions.

    Attributes
    ----------
    options : Adsorptions.Options
        The options for generating reactive complex calculations.
    reactive_site_filter : ReactiveSiteFilter
        The filter applied to determine reactive sites, reactive pairs and trial
        reaction coordinates.
    """

    class Options(FragmentBased.Options):
        """
        The options for adsorptions reactive complex generation.
        """

        class DeactivatedDissOptions(FragmentBased.Options.UnimolDissociationOptions):
            """
                Signal that this inherited option field has no effect on this trial generator.
            """
            __slots__ = ("_deactivated",)

            def __init__(self) -> None:
                self._deactivated = False
                super().__init__()
                self._deactivated = True

            def __setattr__(self, key, value):
                if hasattr(self, "_deactivated") and self._deactivated:
                    raise AttributeError("This option has no effect on this trial generator.")
                super().__setattr__(key, value)

        class DeactivatedAssocOptions(FragmentBased.Options.UnimolAssociationOptions):
            """
                Signal that this inherited option field has no effect on this trial generator.
            """
            __slots__ = ("_deactivated",)

            def __init__(self) -> None:
                self._deactivated = False
                super().__init__()
                self._deactivated = True

            def __setattr__(self, key, value):
                if hasattr(self, "_deactivated") and self._deactivated:
                    raise AttributeError("This option has no effect on this trial generator.")
                super().__setattr__(key, value)

        class AssociationOptions(FragmentBased.Options.BimolAssociationOptions):

            complex_generator: AdsorptionGenerator

            def __init__(self) -> None:
                super().__init__()
                self.complex_generator = AdsorptionGenerator()  # type: ignore

        def __init__(self, parent: Optional[TrialGenerator] = None) -> None:
            super().__init__(parent)
            self.unimolecular_dissociation_options = self.DeactivatedDissOptions()
            """
            UnimolDissociationOptions
                No effect on this trial generator.
            """
            self.unimolecular_association_options = self.DeactivatedAssocOptions()
            """
            UnimolAssociationOptions
                No effect on this trial generator.
            """
            self.bimolecular_association_options = self.AssociationOptions()
            self.bimolecular_association_options.complex_generator.options.extension = [1, 1, 1]  # type: ignore

        bimolecular_association_options: AssociationOptions
        unimolecular_association_options: DeactivatedAssocOptions
        unimolecular_dissociation_options: DeactivatedDissOptions

    options: Options

    def initialize_collections(self, manager: db.Manager) -> None:
        super().initialize_collections(manager)
        self.options.bimolecular_association_options.complex_generator.properties = self._properties

    @_sanity_check_wrapper
    def bimolecular_coordinates(self, structure_list: List[db.Structure],
                                with_exact_settings_check: bool = False,
                                ordering: Optional[Tuple[int, int]] = None) \
            -> List[AdsorptionResult]:
        if ordering is None:
            ordering = self._adsorption_sanity_check(structure_list)
            if ordering is None:
                return []

        # Generate reactive complexes
        ordered_structures = [structure_list[ordering[0]], structure_list[ordering[1]]]

        structure_id_list = [s.id() for s in ordered_structures]
        # If there is a reactive complex calculation for the same structures, return empty list
        selection = {
            "$and": [
                {"job.order": self.options.bimolecular_association_options.job.order},
                {"auxiliaries.lhs": {"$oid": str(structure_id_list[0])}},
                {"auxiliaries.rhs": {"$oid": str(structure_id_list[1])}},
            ] + model_query(self.options.model)
        }
        if not with_exact_settings_check and \
                self._calculations.get_one_calculation(dumps(selection)) is not None:
            return []
        if not len(ordered_structures) == 2:
            raise RuntimeError("Exactly two structures are needed for setting up an adsorption reaction.")
        return self.options.bimolecular_association_options.complex_generator.generate_reactive_complexes(
            ordered_structures[0], ordered_structures[1], self.reactive_site_filter)

    @_sanity_check_wrapper
    def adsorption_reactions(self, structure_list: List[db.Structure], with_exact_settings_check: bool = False) -> None:
        """
        Creates reactive complex calculations corresponding to the bimolecular
        reactions between the structures if there is not already a calculation
        to search for a reaction of the same structures with the same job order.

        Parameters
        ----------
        structure_list : List[db.Structure]
            List of the two structures to be considered.
            The Structures have to be linked to a database.
        """
        ordering = self._adsorption_sanity_check(structure_list)
        if ordering is None:
            return
        ordered_structures = [structure_list[ordering[0]], structure_list[ordering[1]]]
        structure_id_list = [s.id() for s in ordered_structures]

        adsorption_results = self.bimolecular_coordinates(structure_list, with_exact_settings_check, ordering)
        if not adsorption_results:
            return

        charge = sum(structure.get_charge() for structure in structure_list)
        multiplicity = 0  # dummy multiplicity to be corrected by puffin based on wanted multiplicity setting

        if with_exact_settings_check:
            rc_selection = {
                "$and": [
                    {"label": "surface_adsorption_guess"},
                    {"charge": charge},
                ]
            }
            all_complexes = [s.get_atoms().positions for s in self._structures.iterate_structures(dumps(rc_selection))]
        else:
            all_complexes = []

        new_calculation_ids = []
        for result in adsorption_results:
            model = deepcopy(self.options.model)
            model.periodic_boundaries = str(result.pbc)

            # don't use periodic boundaries for comparison, we should always have the same structure in the cell
            if all_complexes and any(np.allclose(result.atoms.positions, rc) for rc in all_complexes):
                continue

            new_structure = db.Structure(db.ID(), self._structures)
            new_structure.create(result.atoms, charge, multiplicity, label=db.Label.SURFACE_ADSORPTION_GUESS,
                                 model=model)
            new_structure.set_comment(f'Adsorption generated from atom {result.adsorbing_atom_name} adsorbing on '
                                      f'{result.site_name}')

            self._set_properties_of_adsorbed_structure(new_structure, result, model)

            reactive_structures = [new_structure.id()] + structure_id_list
            lhs_list = list(set(result.surface_close_atom_indices))
            rhs_list = [len(ordered_structures[0].get_atoms()) +  # shift rhs with lhs atom length
                        result.reactive_index]
            settings = deepcopy(self.options.bimolecular_association_options.job_settings)
            settings["nt_nt_movable_side"] = "rhs"  # avoid strain on surface indices
            cid = self._add_reactive_complex_calculation(
                reactive_structures,
                ReactionType.Associative,
                lhs_list,
                rhs_list,
                self.options.bimolecular_association_options.job,
                settings
            )
            if cid is not None:
                new_calculation_ids.append(cid)
                calculation = db.Calculation(cid, self._calculations)
                calculation.set_auxiliaries({"lhs": structure_id_list[0], "rhs": structure_id_list[1]})
        if new_calculation_ids:
            for s in ordered_structures:
                s.add_calculations(self.get_bimolecular_job_order(), [new_calculation_ids[0]])

    @staticmethod
    def _adsorption_sanity_check(structure_list: List[db.Structure]) -> Optional[Tuple[int, int]]:
        # Check number of compounds
        if len(structure_list) != 2:
            raise RuntimeError("Exactly two structures are needed for setting up an adsorption reaction.")

        surface_index = None
        for i, structure in enumerate(structure_list):
            if PeriodicUtils.is_surface(structure):
                if surface_index is not None:
                    return None
                surface_index = i
        if surface_index is None:
            return None
        ordering = (0, 1) if surface_index == 0 else (1, 0)  # surface structure should be first
        return ordering

    def _set_properties_of_adsorbed_structure(self, structure: db.Structure, result: AdsorptionResult, model: db.Model):

        indices_data = np.array([float(i) for i in result.surface_atom_indices])
        slab = PmgInterface.to_slab(result.slab_dict)

        surf_atoms_prop = db.VectorProperty.make('surface_atom_indices', model,
                                                 indices_data, self._properties)
        slab_property = db.StringProperty.make('slab_dict', self.options.model,
                                               str(result.slab_dict), self._properties)
        formula_prop = db.StringProperty.make('slab_formula', model,
                                              PmgInterface.formula_from_pymatgen_slab(slab),
                                              self._properties)
        true_adsorb_prop = db.BoolProperty.make('true_adsorption', model,
                                                result.true_adsorption, self._properties)

        structure.set_property('surface_atom_indices', surf_atoms_prop.id())
        structure.set_property('slab_dict', slab_property.id())
        structure.set_property('slab_formula', formula_prop.id())
        structure.set_property('true_adsorption', true_adsorb_prop.id())

    def bimolecular_reactions(self, structure_list: List[db.Structure], with_exact_settings_check: bool = False) \
            -> None:
        self.adsorption_reactions(structure_list, with_exact_settings_check)

    def unimolecular_coordinates(self, structure: db.Structure, with_exact_settings_check: bool = False) \
            -> List[Tuple[List[List[Tuple[int, int]]], int]]:
        """
        Returns the unimolecular coordinates of a structure, always empty for this class.
        """
        return []

    def unimolecular_reactions(self, structure: db.Structure, with_exact_settings_check: bool = False) -> None:
        raise NotImplementedError("Adsorptions are only possible as bimolecular reactions")
