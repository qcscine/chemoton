#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
import json
from typing import List, Tuple

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application tests imports
from .... import test_database_setup as db_setup
from ....resources import resources_root_path

# Local application imports
from .....gears.elementary_steps.trial_generator.fragment_based import FragmentBased
from .....gears.elementary_steps.reactive_site_filters import ReactiveSiteFilter


def test_bimol_associations():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_bimol_associations_fragment_based")
    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")

    structure_list = []

    # Add fake data
    model = db.Model("FAKE", "FAKE", "F-AKE")
    rr = resources_root_path()
    for mol in ["hydrogenperoxide", "water"]:
        compound = db.Compound()
        compound.link(compounds)
        compound.create([])
        graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
        structure.set_label(db.Label.USER_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", graph["masm_idx_map"])
        compound.add_structure(structure.id())
        structure.set_aggregate(compound.id())
        structure_list.append(structure)

    # Set up trial generator
    trial_generator = FragmentBased()
    trial_generator.initialize_collections(manager)
    trial_generator.options.model = model
    trial_generator.options.bimolecular_association_options.complex_generator.options.number_rotamers = 1
    trial_generator.options.bimolecular_association_options.complex_generator.options.number_rotamers_two_on_two = 1
    trial_generator.options.bimolecular_association_options.complex_generator.options.multiple_attack_points = False
    trial_generator.options.bimolecular_association_options.consider_diatomic_fragments = True
    trial_generator.options.bimolecular_association_options.max_within_fragment_graph_distance = 1
    # Generate trials
    trial_generator.bimolecular_reactions(structure_list)

    # Checks
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0

    # Expected numbers of trials:
    # atom on atom: 4 * 3 = 12
    # bond of H2O2 with atom of H2O = 3 * 3 = 9
    # atom of H2O2 with bond of H2O = 4*2= 8
    # bond of H2O2 with bond of H2O = 3*2*2 = 6*2 = 12 (second factor 2 because of parallel and antiparallel alignment)
    # sum: 41
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 41

    # Rerun and check that not regenerated
    trial_generator.bimolecular_reactions(structure_list)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 41

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 2
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() == model
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) in (1, 2)  # monoatomic and diatomic fragments
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) in (1, 2)  # monoatomic and diatomic fragments
        assert len(set(lhs)) == len(lhs)
        assert len(set(rhs)) == len(rhs)

    # Rerun with different model
    model2 = db.Model("FAKE2", "", "")
    trial_generator.options.model = model2
    trial_generator.bimolecular_reactions(structure_list)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 82

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 2
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() in (model, model2)
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) in (1, 2)  # monoatomic and diatomic fragments
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) in (1, 2)  # monoatomic and diatomic fragments
        assert len(set(lhs)) == len(lhs)
        assert len(set(rhs)) == len(rhs)
        calculation.wipe()
    for s in structure_list:
        s.clear_calculations("scine_react_complex_nt")

    # Reset model
    trial_generator.options.model = model

    # Tests with filters
    class AtomFilter(ReactiveSiteFilter):
        """
        A reactive site filter implementing only a filter for reactive
        atoms with only H atoms regarded as reactive.
        """

        def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
            reactive_atoms = []
            elements = []
            for struct in structure_list:
                elements += struct.get_atoms().elements
            for i in atom_indices:
                if elements[i] == utils.ElementType.H:
                    reactive_atoms.append(i)
            return reactive_atoms

    trial_generator.reactive_site_filter = AtomFilter()

    # Expected numbers of trials:
    # H atom on H atom: 2 * 2 = 4
    # No H-H bonds
    trial_generator.bimolecular_reactions(structure_list)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 4
    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 2
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() == model
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) == 1
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) == 1  # monoatomic and diatomic fragments
        assert len(set(lhs)) == len(lhs)
        assert len(set(rhs)) == len(rhs)
        calculation.wipe()
    for s in structure_list:
        s.clear_calculations("scine_react_complex_nt")

    class PairFilter(ReactiveSiteFilter):
        """
        A reactive site filter implementing only a filter for atom pairs.
        Only O-O pairs are considered as reactive.
        """

        def filter_atom_pairs(
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
        ) -> List[Tuple[int, int]]:
            reactive_pairs = []
            elements = []
            for struct in structure_list:
                elements += struct.get_atoms().elements
            for i, j in pairs:
                if elements[i] == utils.ElementType.O and elements[j] == utils.ElementType.O:
                    reactive_pairs.append((i, j))
            return reactive_pairs

    trial_generator.reactive_site_filter = PairFilter()

    # Expected numbers of trials:
    # O on O: 1 * 2 = 2
    # O on O-O = 1
    # Sum: 3
    trial_generator.bimolecular_reactions(structure_list)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 3
    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 2
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() == model
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) in (1, 2)
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) in (1, 2)  # monoatomic and diatomic fragments
        assert len(set(lhs)) == len(lhs)
        assert len(set(rhs)) == len(rhs)
        calculation.wipe()
    for s in structure_list:
        s.clear_calculations("scine_react_complex_nt")

    class CoordinateFilter(ReactiveSiteFilter):
        """
        A reactive site filter implementing only a filter for reaction
        coordinate.
        Only O-O bonds combined with an O atom are to be considered as reactive
        """

        def filter_reaction_coordinates(
            self, structure_list: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
        ) -> List[List[Tuple[int, int]]]:
            reactive_coords = []
            elements = []
            for struct in structure_list:
                elements += struct.get_atoms().elements
            for coord in coordinates:
                if len(coord) != 2:
                    # Two reactive pairs wanted
                    continue
                involved_atoms = list(sum(coord, ()))  # flattens coordinate
                if len(set(involved_atoms)) != 3:
                    # Atom on bond
                    continue
                if all(elements[i] == utils.ElementType.O for i in involved_atoms):
                    reactive_coords.append(coord)

            return reactive_coords

    trial_generator.reactive_site_filter = CoordinateFilter()

    # Expected numbers of trials:
    # O on O-O = 1
    trial_generator.bimolecular_reactions(structure_list)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 1
    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 2
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() == model
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) in (1, 2)
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) in (1, 2)  # monoatomic and diatomic fragments
        assert len(set(lhs)) == len(lhs)
        assert len(set(rhs)) == len(rhs)
        calculation.wipe()
    for s in structure_list:
        s.clear_calculations("scine_react_complex_nt")

    class ImpossibleFilter(ReactiveSiteFilter):
        """
        A reactive site filter allowing only H atoms to be reactive and
        O-O reactive pairs, i.e., ultimately no reaction coordinates will be
        generated.
        """

        def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
            reactive_atoms = []
            elements = []
            for struct in structure_list:
                elements += struct.get_atoms().elements
            for i in atom_indices:
                if elements[i] == utils.ElementType.H:
                    reactive_atoms.append(i)
            return reactive_atoms

        def filter_atom_pairs(
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
        ) -> List[Tuple[int, int]]:
            reactive_pairs = []
            elements = []
            for struct in structure_list:
                elements += struct.get_atoms().elements
            for i, j in pairs:
                if elements[i] == utils.ElementType.O and elements[j] == utils.ElementType.O:
                    reactive_pairs.append((i, j))
            return reactive_pairs

    trial_generator.reactive_site_filter = ImpossibleFilter()

    # No trials expected
    trial_generator.bimolecular_reactions(structure_list)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 0

    # Cleaning
    manager.wipe()


def test_unimol_dissociations():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_unimol_dissociations_fragment_based")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")

    # Add fake data
    model = db.Model("FAKE", "FAKE", "F-AKE")
    rr = resources_root_path()
    for mol in ["cyclohexene"]:
        compound = db.Compound()
        compound.link(compounds)
        compound.create([])
        graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", graph["masm_idx_map"])
        structure.set_graph("masm_decision_list", graph["masm_decision_list"])
        compound.add_structure(structure.id())
        structure.set_aggregate(compound.id())

    # Set up trial generator
    trial_generator = FragmentBased()
    trial_generator.initialize_collections(manager)
    trial_generator.options.model = model
    trial_generator.options.unimolecular_dissociation_options.enabled = True
    trial_generator.options.unimolecular_association_options.enabled = False

    # Generate trials
    trial_generator.unimolecular_reactions(structure)

    # Checks
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0

    # Expected number of trials = number of bonds = 16
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 16
    # Rerun and check that not regenerated
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 16

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() == model
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) == 1
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) == 1
        assert len(set(lhs + rhs)) == len(rhs) + len(lhs)
        calculation.wipe()
    structure.clear_calculations("scine_react_complex_nt")

    # Test with filter

    class AtomFilter(ReactiveSiteFilter):
        """
        A reactive site filter allowing only carbon atoms to be reactive
        """

        def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
            reactive_atoms = []
            elements = []
            for struct in structure_list:
                elements += struct.get_atoms().elements
            for i in atom_indices:
                if elements[i] == utils.ElementType.C:
                    reactive_atoms.append(i)
            return reactive_atoms

    trial_generator.reactive_site_filter = AtomFilter()

    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    # All C-C bonds
    assert len(hits) == 6

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() == model
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) == 1
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) == 1
        assert len(set(lhs + rhs)) == len(rhs) + len(lhs)
        calculation.wipe()
    structure.clear_calculations("scine_react_complex_nt")

    class PairFilter(ReactiveSiteFilter):
        """
        A reactive site filter allowing only C-C pairs to be dissociated
        """

        def filter_atom_pairs(
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
        ) -> List[Tuple[int, int]]:
            reactive_pairs = []
            elements = []
            for struct in structure_list:
                elements += struct.get_atoms().elements
            for i, j in pairs:
                if elements[i] == utils.ElementType.C and elements[j] == utils.ElementType.C:
                    reactive_pairs.append((i, j))
            return reactive_pairs

    trial_generator.reactive_site_filter = PairFilter()

    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    # All C-C bonds
    assert len(hits) == 6

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() == model
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) == 1
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) == 1
        assert len(set(lhs + rhs)) == len(rhs) + len(lhs)
        calculation.wipe()
    structure.clear_calculations("scine_react_complex_nt")

    class CoordinateFilter(ReactiveSiteFilter):
        """
        A reactive site filter implementing only a filter for reaction
        coordinate.
        Only C-H bonds are considered as reactive.
        """

        def filter_reaction_coordinates(
            self, structure_list: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
        ) -> List[List[Tuple[int, int]]]:
            reactive_coords = []
            elements = []
            for struct in structure_list:
                elements += struct.get_atoms().elements
            for coord in coordinates:
                pair_elements = (elements[coord[0][0]], elements[coord[0][1]])
                if utils.ElementType.H in pair_elements and utils.ElementType.C in pair_elements:
                    reactive_coords.append(coord)
            return reactive_coords

    trial_generator.reactive_site_filter = CoordinateFilter()
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    # All C-H bonds
    assert len(hits) == 10

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() == model
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) == 1
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) == 1
        assert len(set(lhs + rhs)) == len(rhs) + len(lhs)
        calculation.wipe()
    structure.clear_calculations("scine_react_complex_nt")

    class CombinedFilter(ReactiveSiteFilter):
        """
        A reactive site filter allowing for C atoms and C-C bonds to be reactive
        """

        def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
            reactive_atoms = []
            elements = []
            for struct in structure_list:
                elements += struct.get_atoms().elements
            for i in atom_indices:
                if elements[i] == utils.ElementType.C:
                    reactive_atoms.append(i)
            return reactive_atoms

        def filter_atom_pairs(
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
        ) -> List[Tuple[int, int]]:
            reactive_pairs = []
            elements = []
            for struct in structure_list:
                elements += struct.get_atoms().elements
            for i, j in pairs:
                if elements[i] == utils.ElementType.C and elements[j] == utils.ElementType.C:
                    reactive_pairs.append((i, j))
            return reactive_pairs

    trial_generator.reactive_site_filter = CombinedFilter()
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    # All C-C bonds
    assert len(hits) == 6

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() == model
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) == 1
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) == 1
        assert len(set(lhs + rhs)) == len(rhs) + len(lhs)

    # Rerun with different model
    model2 = db.Model("FAKE2", "", "")
    trial_generator.options.model = model2
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 12

    # Cleaning
    manager.wipe()


def test_unimol_associations_atom_on_atom():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_unimol_associations_fragment_based")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")

    # Add fake data
    model = db.Model("FAKE", "FAKE", "F-AKE")
    rr = resources_root_path()
    for mol in ["cyclohexene"]:
        compound = db.Compound()
        compound.link(compounds)
        compound.create([])
        graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", graph["masm_idx_map"])
        structure.set_graph("masm_decision_list", graph["masm_decision_list"])
        compound.add_structure(structure.id())
        structure.set_aggregate(compound.id())

    # Set up trial generator
    trial_generator = FragmentBased()
    trial_generator.initialize_collections(manager)
    trial_generator.options.model = model
    trial_generator.options.unimolecular_dissociation_options.enabled = False
    trial_generator.options.unimolecular_association_options.enabled = True
    trial_generator.options.unimolecular_association_options.consider_diatomic_fragments = False

    # Generate trials
    trial_generator.unimolecular_reactions(structure)

    # Checks
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 104

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() == model
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) == 1
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) == 1
        assert len(set(lhs + rhs)) == len(rhs) + len(lhs)

    # Run a second time
    trial_generator.unimolecular_reactions(structure)

    # Check again
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 104

    # Rerun with different model
    model2 = db.Model("FAKE2", "", "")
    trial_generator.options.model = model2
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 208

    # Cleaning
    manager.wipe()


def test_unimol_associations():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_unimol_associations_fragment_based")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")

    # Add fake data
    model = db.Model("FAKE", "FAKE", "F-AKE")
    rr = resources_root_path()
    for mol in ["hydrogenperoxide"]:
        compound = db.Compound()
        compound.link(compounds)
        compound.create([])
        graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", graph["masm_idx_map"])
        structure.set_graph("masm_decision_list", graph["masm_decision_list"])
        compound.add_structure(structure.id())
        structure.set_aggregate(compound.id())

    # Set up trial generator
    trial_generator = FragmentBased()
    trial_generator.initialize_collections(manager)
    trial_generator.options.model = model
    trial_generator.options.unimolecular_dissociation_options.enabled = False
    trial_generator.options.unimolecular_association_options.enabled = True
    trial_generator.options.unimolecular_association_options.min_inter_fragment_graph_distance = 2
    trial_generator.options.unimolecular_association_options.max_inter_fragment_graph_distance = 999
    # Enable bond on bond
    trial_generator.options.unimolecular_association_options.consider_diatomic_fragments = True
    trial_generator.options.unimolecular_association_options.max_within_fragment_graph_distance = 1

    # Generate trials
    trial_generator.unimolecular_reactions(structure)

    # Checks
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0

    # Expected number of hits for H1-O1-O2-H2
    # atom on atom:  3 (H1 <-> H2, H1 <-> O2, H2 <-> O1)
    # atom on bond:  2 (H1 <-> O2-H2, H2 <-> O1-H1)
    # bond on bond:  0 (H1-O1 and H2-O2 do not conform to minimum interfragment distance due to O-O bond)
    # Sum:           5
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 5

    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    # Check again
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 5

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() == model
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) in (1, 2)
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) in (1, 2)
        assert len(set(lhs + rhs)) == len(rhs) + len(lhs)
        calculation.wipe()
    structure.clear_calculations("scine_react_complex_nt")

    # Test with filter
    class AtomFilter(ReactiveSiteFilter):
        """
        A reactive site filter allowing only hydrogen atoms to be reactive
        """

        def filter_atoms(self, structure_list: List[db.Structure], atom_indices: List[int]) -> List[int]:
            reactive_atoms = []
            elements = []
            for struct in structure_list:
                elements += struct.get_atoms().elements
            for i in atom_indices:
                if elements[i] == utils.ElementType.H:
                    reactive_atoms.append(i)
            return reactive_atoms

    trial_generator.reactive_site_filter = AtomFilter()
    trial_generator.unimolecular_reactions(structure)

    # Only H on H allowed
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 1
    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() == model
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) == 1
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) == 1
        assert len(set(lhs + rhs)) == len(rhs) + len(lhs)
        calculation.wipe()
    structure.clear_calculations("scine_react_complex_nt")

    class PairFilter(ReactiveSiteFilter):
        """
        A reactive site filter allowing only pairs with at least one oxygen
        atom to be reactive
        """

        def filter_atom_pairs(
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
        ) -> List[Tuple[int, int]]:
            reactive_pairs = []
            elements = []
            for struct in structure_list:
                elements += struct.get_atoms().elements
            for pair in pairs:
                if utils.ElementType.O in (elements[pair[0]], elements[pair[1]]):
                    reactive_pairs.append(pair)
            return reactive_pairs

    trial_generator.reactive_site_filter = PairFilter()

    trial_generator.unimolecular_reactions(structure)
    # Expected number of hits for H1-O1-O2-H2
    # atom on atom:  2 (H1 <-> O2, H2 <-> O1)
    # atom on bond:  0 (would be H on H-O in brute force approach i.e. there is always an H-H reactive pair)
    # bond on bond:  0 (H1-O1 and H2-O2 do not conform to minimum interfragment distance due to O-O bond)
    # Sum:           2
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 2
    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() == model
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) == 1
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) == 1
        assert len(set(lhs + rhs)) == len(rhs) + len(lhs)
        calculation.wipe()
    structure.clear_calculations("scine_react_complex_nt")

    class CoordinateFilter(ReactiveSiteFilter):
        """
        A reactive site filter allowing only coordinates with three distinct
        atoms
        """

        def filter_reaction_coordinates(
            self, structure_list: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
        ) -> List[List[Tuple[int, int]]]:
            reactive_coords = []
            elements = []
            for struct in structure_list:
                elements += struct.get_atoms().elements
            for coord in coordinates:
                involved_atoms = list(sum(coord, ()))  # flattens coordinate
                if len(set(involved_atoms)) == 3:
                    # Atom on bond
                    reactive_coords.append(coord)
            return reactive_coords

    trial_generator.reactive_site_filter = CoordinateFilter()

    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    # Expected number of hits for H1-O1-O2-H2
    # # atom on bond: 2 (H1 <-> O2-H2, H2 <-> O1-H1)
    # Sum:           2
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 2
    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt"
        assert calculation.get_model() == model
        lhs = calculation.get_setting("nt_nt_lhs_list")
        assert len(lhs) in (1, 2)
        rhs = calculation.get_setting("nt_nt_rhs_list")
        assert len(rhs) in (1, 2)
        assert len(set(lhs + rhs)) == len(rhs) + len(lhs)
        assert len(set(lhs + rhs)) == 3

    # Cleaning
    manager.wipe()
