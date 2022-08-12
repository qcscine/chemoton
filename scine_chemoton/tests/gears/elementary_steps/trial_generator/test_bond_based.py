#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
import json

# Third party imports
import scine_database as db

# Local application tests imports
from .... import test_database_setup as db_setup
from ....resources import resources_root_path

# Local application imports
from .....gears.elementary_steps.trial_generator.bond_based import BondBased


def test_bimol():
    """
    Tests whether bimolecular elementary step trials are set up correctly
    with the bond-based trial generator
    """
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_bimol_bond_based")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")

    # Add fake data
    model = db.Model("FAKE", "FAKE", "F-AKE")
    rr = resources_root_path()
    structure_list = []
    for mol in ["hydrogenperoxide", "water"]:
        graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", graph["masm_idx_map"])
        structure.set_graph("masm_decision_list", graph["masm_decision_list"])
        compound = db.Compound()
        compound.link(compounds)
        compound.create([structure.id()])
        structure.set_aggregate(compound.id())
        structure_list.append(structure)

    # Setup trial generator with impossible settings:
    # Cannot fulfil all minimum requirements and the bond modification boundaries simultaneously
    trial_generator = BondBased()
    trial_generator.initialize_collections(manager)
    trial_generator.options.model = model
    trial_generator.options.bimolecular.min_bond_modifications = 1
    trial_generator.options.bimolecular.max_bond_modifications = 2
    trial_generator.options.bimolecular.min_inter_bond_formations = 1
    trial_generator.options.bimolecular.max_inter_bond_formations = 2
    trial_generator.options.bimolecular.min_intra_bond_formations = 1
    trial_generator.options.bimolecular.max_intra_bond_formations = 1
    trial_generator.options.bimolecular.min_bond_dissociations = 1
    trial_generator.options.bimolecular.max_bond_dissociations = 1
    trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = False
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers = 1
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers_two_on_two = 1
    # Generate trials
    trial_generator.bimolecular_reactions(structure_list)

    # Checks
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 0

    # Setup trial coordinator with only one or two inter coordinates
    trial_generator.options.bimolecular.min_bond_modifications = 1
    trial_generator.options.bimolecular.max_bond_modifications = 2
    trial_generator.options.bimolecular.min_inter_bond_formations = 1
    trial_generator.options.bimolecular.max_inter_bond_formations = 2
    trial_generator.options.bimolecular.min_intra_bond_formations = 0
    trial_generator.options.bimolecular.max_intra_bond_formations = 0
    trial_generator.options.bimolecular.min_bond_dissociations = 0
    trial_generator.options.bimolecular.max_bond_dissociations = 0
    trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = False
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers = 1
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers_two_on_two = 1
    # Generate trials
    trial_generator.bimolecular_reactions(structure_list)

    # Expected number of trials
    # One intercoordinate: 4 * 3 = 12
    # Two intercoordinates: 12 choose 2 = 66
    # Sum: 78
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 78

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 2
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_model() == model
        assert calculation.get_job().order == "scine_react_complex_nt2"
        n_formed = len(calculation.get_setting("nt_nt_associations"))
        assert n_formed in (2, 4)  # Two entries per pair
        n_diss = len(calculation.get_setting("nt_nt_dissociations"))
        assert n_diss == 0
        assert 2 <= n_formed + n_diss <= 4
        calculation.wipe()

    # Setup trial coordinator with only one inter coordinate or one inter and one dissociation
    trial_generator.options.bimolecular.min_bond_modifications = 1
    trial_generator.options.bimolecular.max_bond_modifications = 2
    trial_generator.options.bimolecular.min_inter_bond_formations = 1
    trial_generator.options.bimolecular.max_inter_bond_formations = 1
    trial_generator.options.bimolecular.min_intra_bond_formations = 0
    trial_generator.options.bimolecular.max_intra_bond_formations = 0
    trial_generator.options.bimolecular.min_bond_dissociations = 0
    trial_generator.options.bimolecular.max_bond_dissociations = 1
    trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = False
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers = 1
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers_two_on_two = 1
    # Generate trials
    trial_generator.bimolecular_reactions(structure_list)

    # Expected number of trials
    # One intercoordinate: 4 * 3 = 12
    # One inter and one diss = 12 * 5 = 60
    # Sum: 72
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 72

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 2
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_model() == model
        assert calculation.get_job().order == "scine_react_complex_nt2"
        n_formed = len(calculation.get_setting("nt_nt_associations"))
        assert n_formed == 2  # Two entries per pair
        n_diss = len(calculation.get_setting("nt_nt_dissociations"))
        assert n_diss in (0, 2)
        assert 2 <= n_formed + n_diss <= 4
        calculation.wipe()

    # Setup trial coordinator with only one inter and one associative intra
    trial_generator.options.bimolecular.min_bond_modifications = 1  # Not to be reached
    trial_generator.options.bimolecular.max_bond_modifications = 2
    trial_generator.options.bimolecular.min_inter_bond_formations = 1
    trial_generator.options.bimolecular.max_inter_bond_formations = 1
    trial_generator.options.bimolecular.min_intra_bond_formations = 1
    trial_generator.options.bimolecular.max_intra_bond_formations = 1
    trial_generator.options.bimolecular.min_bond_dissociations = 0
    trial_generator.options.bimolecular.max_bond_dissociations = 0
    trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = False
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers = 1
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers_two_on_two = 1
    # Generate trials
    trial_generator.bimolecular_reactions(structure_list)

    # Expected number of trials
    # Number or intermolecular pairs= 4 * 3 = 12
    # Number of intramolecular associative pairs = 4
    # 4*12 = 48 combinations
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 48

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 2
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_model() == model
        assert calculation.get_job().order == "scine_react_complex_nt2"
        n_formed = len(calculation.get_setting("nt_nt_associations"))
        assert n_formed == 4  # Two entries per pair
        n_diss = len(calculation.get_setting("nt_nt_dissociations"))
        assert n_diss == 0
        assert 2 <= n_formed + n_diss <= 4
        calculation.wipe()

    # Setup  trial generator with mixed inter, intra associative and dissociative coordinates
    trial_generator.options.bimolecular.min_bond_modifications = 1
    trial_generator.options.bimolecular.max_bond_modifications = 2
    trial_generator.options.bimolecular.min_inter_bond_formations = 1
    trial_generator.options.bimolecular.max_inter_bond_formations = 2
    trial_generator.options.bimolecular.min_intra_bond_formations = 0
    trial_generator.options.bimolecular.max_intra_bond_formations = 3  # Not to be reached due to max bond modifications
    trial_generator.options.bimolecular.min_bond_dissociations = 0
    trial_generator.options.bimolecular.max_bond_dissociations = 3  # Not to be reached due to max bond modifications
    trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = False
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers = 1
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers_two_on_two = 1
    # Generate trials
    trial_generator.bimolecular_reactions(structure_list)

    # Checks
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0

    # Expected number of hits:
    # number of intermolecular pairs = n_atoms(H2O) * n_atoms(H2O2) = 3*4 = 12
    # number of pairs of pairs = n_pairs over 2 = 12 over 2 = 66
    # Number of bonds: 3 (H2O2) + 2 (H2O) = 5
    # Number of intramolecular unbound pairs: 3 (H2O2) + 1 (H2O) = 4
    # Expected number of hits:
    # Two intermol coordinates: 66
    # One intermol coordinate: 12
    # One intermol + one diss: 12*5 = 60
    # One intermol + one intramol formation = 12*4 = 48
    # Sum: 186

    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 186

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 2
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_model() == model
        assert calculation.get_job().order == "scine_react_complex_nt2"
        n_formed = len(calculation.get_setting("nt_nt_associations"))
        assert n_formed in (2, 4)  # Two entries per pair
        n_diss = len(calculation.get_setting("nt_nt_dissociations"))
        assert n_diss in (0, 2)  # Two entries per pair
        assert 2 <= n_formed + n_diss <= 4

    # Run a second time
    trial_generator.bimolecular_reactions(structure_list)

    # Check again
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 186

    # Rerun with a different model
    model2 = db.Model("FAKE2", "", "")
    trial_generator.options.model = model2
    trial_generator.bimolecular_reactions(structure_list)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 372

    # Cleaning
    manager.wipe()


def test_unimol_dissociations():
    """
    Tests whether unimolecular elementary step trials containing only
    dissociations are set up correctly with the bond-based trial generator
    """
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_unimol_dissociations_bond_based")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")

    # Add fake data
    model = db.Model("FAKE", "FAKE", "F-AKE")
    rr = resources_root_path()
    for mol in ["cyclohexene"]:
        graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", graph["masm_idx_map"])
        structure.set_graph("masm_decision_list", graph["masm_decision_list"])
        compound = db.Compound()
        compound.link(compounds)
        compound.create([structure.id()])
        structure.set_aggregate(compound.id())

    # Setup trial generator with settings that cannot be fulfilled
    trial_generator = BondBased()
    trial_generator.initialize_collections(manager)
    trial_generator.options.model = model
    trial_generator.options.unimolecular.min_bond_modifications = 1
    trial_generator.options.unimolecular.max_bond_modifications = 1
    trial_generator.options.unimolecular.min_bond_dissociations = 2
    trial_generator.options.unimolecular.min_bond_formations = 2
    # Generate trials
    trial_generator.unimolecular_reactions(structure)

    # Checks
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 0

    # Exactly one dissociation at once
    trial_generator.options.unimolecular.min_bond_modifications = 1
    trial_generator.options.unimolecular.max_bond_modifications = 1
    trial_generator.options.unimolecular.min_bond_dissociations = 1
    trial_generator.options.unimolecular.max_bond_dissociations = 1
    trial_generator.options.unimolecular.min_bond_formations = 0
    trial_generator.options.unimolecular.max_bond_formations = 0

    trial_generator.unimolecular_reactions(structure)
    # Expected number of hits = number of bonds = 16
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 16

    # Run a second time
    trial_generator.unimolecular_reactions(structure)

    # Check again
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 16

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt2"
        assert calculation.get_model() == model
        n_formed = len(calculation.get_setting("nt_nt_associations"))
        assert n_formed == 0
        n_diss = len(calculation.get_setting("nt_nt_dissociations"))
        assert n_diss == 2  # Two entries per pair
        assert 2 <= n_formed + n_diss <= 2
        # Cleaning
        calculation.wipe()

    # Try exactly two dissociations at once
    trial_generator.options.unimolecular.min_bond_modifications = 1  # Not to be reached
    trial_generator.options.unimolecular.max_bond_modifications = 2
    trial_generator.options.unimolecular.min_bond_dissociations = 2
    trial_generator.options.unimolecular.max_bond_dissociations = 2
    trial_generator.options.unimolecular.min_bond_formations = 0
    trial_generator.options.unimolecular.max_bond_formations = 0

    # Expected number of hits = number of bonds choose 2 = 16 choose 2 = 120
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 120

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt2"
        n_formed = len(calculation.get_setting("nt_nt_associations"))
        assert n_formed == 0
        n_diss = len(calculation.get_setting("nt_nt_dissociations"))
        assert n_diss == 4  # Two entries per pair
        assert 2 <= n_formed + n_diss <= 4
        # Cleaning
        calculation.wipe()

    # Try one or two dissociations at once
    trial_generator.options.unimolecular.min_bond_modifications = 1
    trial_generator.options.unimolecular.max_bond_modifications = 3  # Not to be reached
    trial_generator.options.unimolecular.min_bond_dissociations = 1
    trial_generator.options.unimolecular.max_bond_dissociations = 2
    trial_generator.options.unimolecular.min_bond_formations = 0
    trial_generator.options.unimolecular.max_bond_formations = 0

    # Expected number of hits
    # Single dissociations = number of bonds = 16
    # Double dissociations = number of bonds choose 2 = 16 choose 2 = 120
    # Sum: 136
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 136

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt2"
        assert calculation.get_model() == model
        n_formed = len(calculation.get_setting("nt_nt_associations"))
        assert n_formed == 0
        n_diss = len(calculation.get_setting("nt_nt_dissociations"))
        assert n_diss in (2, 4)  # Two entries per pair
        assert 2 <= n_formed + n_diss <= 6

    # Cleaning
    manager.wipe()


def test_unimol_associations():
    """
    Tests whether unimolecular elementary step trials containing only
    associations are set up correctly with the bond-based trial generator
    """
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_unimol_associations_bond_based")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")

    # Add fake data
    model = db.Model("FAKE", "FAKE", "F-AKE")
    rr = resources_root_path()
    for mol in ["hydrogenperoxide"]:
        graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", graph["masm_idx_map"])
        structure.set_graph("masm_decision_list", graph["masm_decision_list"])
        compound = db.Compound()
        compound.link(compounds)
        compound.create([structure.id()])
        structure.set_aggregate(compound.id())

    # Setup trial generator with settings that cannot be fulfilled
    trial_generator = BondBased()
    trial_generator.initialize_collections(manager)
    trial_generator.options.model = model
    trial_generator.options.unimolecular.min_bond_modifications = 2
    trial_generator.options.unimolecular.max_bond_modifications = 1
    trial_generator.options.unimolecular.min_bond_dissociations = 0
    trial_generator.options.unimolecular.max_bond_dissociations = 0
    trial_generator.options.unimolecular.min_bond_formations = 0
    trial_generator.options.unimolecular.max_bond_formations = 1
    # Generate trials
    trial_generator.unimolecular_reactions(structure)

    # Checks
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 0

    # Test exactly one association
    trial_generator.options.unimolecular.min_bond_modifications = 1
    trial_generator.options.unimolecular.max_bond_modifications = 1
    trial_generator.options.unimolecular.min_bond_dissociations = 0
    trial_generator.options.unimolecular.max_bond_dissociations = 0
    trial_generator.options.unimolecular.min_bond_formations = 1
    trial_generator.options.unimolecular.max_bond_formations = 1

    trial_generator.unimolecular_reactions(structure)
    # Expected number of hits:
    # Number of pairs: 4 choose 2 = 6
    # Number of bonds: 3
    # Number of unbound pairs = expected number of hits = 3

    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 3

    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 3

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_model() == model
        assert calculation.get_job().order == "scine_react_complex_nt2"
        n_formed = len(calculation.get_setting("nt_nt_associations"))
        assert n_formed == 2  # 2 entries per pair
        n_diss = len(calculation.get_setting("nt_nt_dissociations"))
        assert n_diss == 0
        assert 2 <= n_formed + n_diss <= 2
        # Clean up
        calculation.wipe()

    # Test exactly two associations
    trial_generator.options.unimolecular.min_bond_modifications = 2
    trial_generator.options.unimolecular.max_bond_modifications = 2
    trial_generator.options.unimolecular.min_bond_dissociations = 0
    trial_generator.options.unimolecular.max_bond_dissociations = 0
    trial_generator.options.unimolecular.min_bond_formations = 1  # Not to be reached
    trial_generator.options.unimolecular.max_bond_formations = 2

    trial_generator.unimolecular_reactions(structure)
    # Expected number of hits:
    # Number of unbound pairs choose 2 = 3 choose 2 = 3

    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 3

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_model() == model
        assert calculation.get_job().order == "scine_react_complex_nt2"
        n_formed = len(calculation.get_setting("nt_nt_associations"))
        assert n_formed == 4  # 2 entries per pair
        n_diss = len(calculation.get_setting("nt_nt_dissociations"))
        assert n_diss == 0
        assert 4 <= n_formed + n_diss <= 4
        # Clean up
        calculation.wipe()

    # Test one or two associations
    trial_generator.options.unimolecular.min_bond_modifications = 1
    trial_generator.options.unimolecular.max_bond_modifications = 2
    trial_generator.options.unimolecular.min_bond_dissociations = 0
    trial_generator.options.unimolecular.max_bond_dissociations = 0
    trial_generator.options.unimolecular.min_bond_formations = 1  # Not to be reached
    trial_generator.options.unimolecular.max_bond_formations = 2

    trial_generator.unimolecular_reactions(structure)
    # Expected number of hits:
    # Number of unbound pairs = 3
    # Number of unbound pairs choose 2 = 3 choose 2 = 3
    # Sum: 6

    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 6

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_model() == model
        assert calculation.get_job().order == "scine_react_complex_nt2"
        n_formed = len(calculation.get_setting("nt_nt_associations"))
        assert n_formed in (2, 4)  # 2 entries per pair
        n_diss = len(calculation.get_setting("nt_nt_dissociations"))
        assert n_diss == 0
        assert 2 <= n_formed + n_diss <= 4

    # Cleaning
    manager.wipe()


def test_unimol_mixed():
    """
    Tests whether unimolecular elementary step trials containing only
    dissociations and associations are set up correctly with the bond-based
    trial generator
    """
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_unimol_mixed_bond_based")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")

    # Add fake data
    model = db.Model("FAKE", "FAKE", "F-AKE")
    rr = resources_root_path()
    for mol in ["hydrogenperoxide"]:
        graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", graph["masm_idx_map"])
        structure.set_graph("masm_decision_list", graph["masm_decision_list"])
        compound = db.Compound()
        compound.link(compounds)
        compound.create([structure.id()])
        structure.set_aggregate(compound.id())

    # Setup trial generator for either one dissociation or one association
    trial_generator = BondBased()
    trial_generator.initialize_collections(manager)
    trial_generator.options.model = model
    trial_generator.options.unimolecular.min_bond_modifications = 1
    trial_generator.options.unimolecular.max_bond_modifications = 1
    trial_generator.options.unimolecular.min_bond_dissociations = 0
    trial_generator.options.unimolecular.max_bond_dissociations = 1
    trial_generator.options.unimolecular.min_bond_formations = 0
    trial_generator.options.unimolecular.max_bond_formations = 1
    # Generate trials
    trial_generator.unimolecular_reactions(structure)

    # Expected number of hits:
    # Number of bond: 3
    # Number of unbound pairs: 3
    # Sum: 6
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 6

    trial_generator.unimolecular_reactions(structure)

    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 6

    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 6

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_model() == model
        assert calculation.get_job().order == "scine_react_complex_nt2"
        n_formed = len(calculation.get_setting("nt_nt_associations"))
        assert n_formed in (0, 2)  # 2 entries per pair
        n_diss = len(calculation.get_setting("nt_nt_dissociations"))
        assert n_diss in (0, 2)
        assert 2 <= n_formed + n_diss <= 2
        # Clean up
        calculation.wipe()

    # Setup trial generator up to 3 intramolecular coordinates of any type
    trial_generator = BondBased()
    trial_generator.initialize_collections(manager)
    trial_generator.options.model = model
    trial_generator.options.unimolecular.min_bond_modifications = 1
    trial_generator.options.unimolecular.max_bond_modifications = 3
    trial_generator.options.unimolecular.min_bond_dissociations = 0
    trial_generator.options.unimolecular.max_bond_dissociations = 3
    trial_generator.options.unimolecular.min_bond_formations = 0
    trial_generator.options.unimolecular.max_bond_formations = 3
    # Generate trials
    trial_generator.unimolecular_reactions(structure)

    # Expected number of hits:
    # Number of pairs (bound + unbound) = one modification: 6
    # Two modifications: 6 choose 2: 15
    # Three modifications: 6 choose 3: 20
    # Sum: 41

    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 41

    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 41

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_model() == model
        assert calculation.get_job().order == "scine_react_complex_nt2"
        n_formed = len(calculation.get_setting("nt_nt_associations"))
        assert n_formed in (0, 2, 4, 6)  # 2 entries per pair
        n_diss = len(calculation.get_setting("nt_nt_dissociations"))
        assert n_diss in (0, 2, 4, 6)  # 2 entries per pair
        assert 2 <= n_formed + n_diss <= 6

    # Rerun with a different model
    model2 = db.Model("FAKE2", "", "")
    trial_generator.options.model = model2
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 82

    # Cleaning
    manager.wipe()


def test_estimate_n_unimolecular_trials():
    """
    Tests whether the number of trial reactive coordinates forecasted for
    unimolecular reactions is correct
    """
    rr = resources_root_path()
    structure_file = os.path.join(rr, "hydrogenperoxide.xyz")
    trial_generator = BondBased()

    # Options that should not allow any trials
    trial_generator.options.unimolecular.min_bond_modifications = 2
    trial_generator.options.unimolecular.max_bond_modifications = 1
    n_trials = trial_generator.estimate_n_unimolecular_trials(structure_file)
    assert n_trials == 0

    trial_generator.options.unimolecular.min_bond_modifications = 3
    trial_generator.options.unimolecular.max_bond_modifications = 4
    trial_generator.options.unimolecular.min_bond_dissociations = 0
    trial_generator.options.unimolecular.max_bond_dissociations = 1
    trial_generator.options.unimolecular.min_bond_formations = 0
    trial_generator.options.unimolecular.max_bond_formations = 1
    n_trials = trial_generator.estimate_n_unimolecular_trials(structure_file)
    assert n_trials == 0

    # Test legitimate options with all pairs
    # Associations only
    trial_generator.options.unimolecular.min_bond_modifications = 1
    trial_generator.options.unimolecular.max_bond_modifications = 3
    trial_generator.options.unimolecular.min_bond_dissociations = 0
    trial_generator.options.unimolecular.max_bond_dissociations = 0
    trial_generator.options.unimolecular.min_bond_formations = 0
    trial_generator.options.unimolecular.max_bond_formations = 1
    # Expected number of trials
    # 1 form: 3
    # Sum: 3
    n_trials = trial_generator.estimate_n_unimolecular_trials(structure_file)
    assert n_trials == 3

    # Dissociations only
    trial_generator.options.unimolecular.min_bond_modifications = 1
    trial_generator.options.unimolecular.max_bond_modifications = 3
    trial_generator.options.unimolecular.min_bond_dissociations = 0
    trial_generator.options.unimolecular.max_bond_dissociations = 3
    trial_generator.options.unimolecular.min_bond_formations = 0
    trial_generator.options.unimolecular.max_bond_formations = 0
    # Expected number of trials
    # 1 diss: 3
    # 2 diss: 3
    # 3 diss: 1
    # Sum: 7
    n_trials = trial_generator.estimate_n_unimolecular_trials(structure_file)
    assert n_trials == 7

    # Mixed associations and dissociations
    trial_generator.options.unimolecular.min_bond_modifications = 1
    trial_generator.options.unimolecular.max_bond_modifications = 3
    trial_generator.options.unimolecular.min_bond_dissociations = 0
    trial_generator.options.unimolecular.max_bond_dissociations = 1
    trial_generator.options.unimolecular.min_bond_formations = 0
    trial_generator.options.unimolecular.max_bond_formations = 2
    # Expected number of trials
    # 1 diss: 3
    # 1 form: 3
    # 2 form: 3
    # 1 form + 1 diss = 3*3 = 9
    # 2 form + 1 diss = 3*3 = 9
    # Sum: 27
    n_trials = trial_generator.estimate_n_unimolecular_trials(structure_file)
    assert n_trials == 27

    # Test with limited numbers of reactive pairs
    # Limit number of bound pairs to consider for dissociations to one
    # Expected number of trials
    # 1 diss: 1
    # 1 form: 3
    # 2 form: 3
    # 1 form + 1 diss = 3*1 = 3
    # 2 form + 1 diss = 3*1 = 3
    # Sum: 13
    n_trials = trial_generator.estimate_n_unimolecular_trials(structure_file, n_reactive_bound_pairs=1)
    assert n_trials == 13

    # Limit number of unbound pairs to consider for bond formations to one
    # Expected number of trials
    # 1 diss: 3
    # 1 form: 1
    # 2 form: 0
    # 1 form + 1 diss = 1*3 = 3
    # 2 form + 1 diss = 0*3 = 0
    # Sum: 7
    n_trials = trial_generator.estimate_n_unimolecular_trials(structure_file, n_reactive_unbound_pairs=1)
    assert n_trials == 7

    # No reactive pairs
    n_trials = trial_generator.estimate_n_unimolecular_trials(
        structure_file, n_reactive_bound_pairs=0, n_reactive_unbound_pairs=0
    )
    assert n_trials == 0


def test_estimate_n_bimolecular_trials():
    """
    Tests whether the number of trial reactive coordinates forecasted for
    unimolecular reactions is correct
    """
    rr = resources_root_path()
    structure_file1 = os.path.join(rr, "hydrogenperoxide.xyz")
    structure_file2 = os.path.join(rr, "water.xyz")
    trial_generator = BondBased()

    # Options that should not allow any trials
    trial_generator.options.bimolecular.min_bond_modifications = 2
    trial_generator.options.bimolecular.max_bond_modifications = 1
    n_trials = trial_generator.estimate_n_bimolecular_trials(structure_file1, structure_file2)
    assert n_trials == 0

    trial_generator.options.bimolecular.min_bond_modifications = 1
    trial_generator.options.bimolecular.max_bond_modifications = 2
    trial_generator.options.bimolecular.min_inter_bond_formations = 0
    trial_generator.options.bimolecular.max_inter_bond_formations = 0
    n_trials = trial_generator.estimate_n_bimolecular_trials(structure_file1, structure_file2)
    assert n_trials == 0

    trial_generator.options.bimolecular.min_bond_modifications = 2
    trial_generator.options.bimolecular.max_bond_modifications = 1
    n_trials = trial_generator.estimate_n_bimolecular_trials(structure_file1, structure_file2)
    assert n_trials == 0

    trial_generator.options.bimolecular.min_bond_modifications = 1
    trial_generator.options.bimolecular.max_bond_modifications = 1
    trial_generator.options.bimolecular.min_inter_bond_formations = 0
    trial_generator.options.bimolecular.max_inter_bond_formations = 1
    trial_generator.options.bimolecular.min_intra_bond_formations = 1
    n_trials = trial_generator.estimate_n_bimolecular_trials(structure_file1, structure_file2)
    assert n_trials == 0

    # Test legitimate options with all pairs
    # One or two intermolecular bond formations
    trial_generator.options.bimolecular.min_bond_modifications = 1
    trial_generator.options.bimolecular.max_bond_modifications = 2
    trial_generator.options.bimolecular.min_inter_bond_formations = 1
    trial_generator.options.bimolecular.max_inter_bond_formations = 2
    trial_generator.options.bimolecular.min_intra_bond_formations = 0
    trial_generator.options.bimolecular.max_intra_bond_formations = 0
    trial_generator.options.bimolecular.min_bond_dissociations = 0
    trial_generator.options.bimolecular.max_bond_dissociations = 0
    trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = False
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers = 1
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers_two_on_two = 1
    # Expected number of trials
    # One intercoordinate: 4 * 3 = 12
    # Two intercoordinates: 12 choose 2 = 66
    # Sum: 78
    n_trials = trial_generator.estimate_n_bimolecular_trials(structure_file1, structure_file2)
    assert n_trials == 78

    # Setup trial coordinator with only one inter coordinate or one inter and one dissociation
    trial_generator.options.bimolecular.min_bond_modifications = 1
    trial_generator.options.bimolecular.max_bond_modifications = 2
    trial_generator.options.bimolecular.min_inter_bond_formations = 1
    trial_generator.options.bimolecular.max_inter_bond_formations = 1
    trial_generator.options.bimolecular.min_intra_bond_formations = 0
    trial_generator.options.bimolecular.max_intra_bond_formations = 0
    trial_generator.options.bimolecular.min_bond_dissociations = 0
    trial_generator.options.bimolecular.max_bond_dissociations = 1
    trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = False
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers = 1
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers_two_on_two = 1
    # Expected number of trials
    # One intercoordinate: 4 * 3 = 12
    # One inter and one diss = 12 * 5 = 60
    # Sum: 72
    n_trials = trial_generator.estimate_n_bimolecular_trials(structure_file1, structure_file2, attack_points_per_site=2)
    assert n_trials == 72

    # Mixed inter, intra and dissociative
    trial_generator.options.bimolecular.min_bond_modifications = 1
    trial_generator.options.bimolecular.max_bond_modifications = 2
    trial_generator.options.bimolecular.min_inter_bond_formations = 1
    trial_generator.options.bimolecular.max_inter_bond_formations = 2
    trial_generator.options.bimolecular.min_intra_bond_formations = 0
    trial_generator.options.bimolecular.max_intra_bond_formations = 3  # Not to be reached due to max bond modifications
    trial_generator.options.bimolecular.min_bond_dissociations = 0
    trial_generator.options.bimolecular.max_bond_dissociations = 3  # Not to be reached due to max bond modifications
    trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = False
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers = 1
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers_two_on_two = 1
    # Expected number of hits:
    # number of intermolecular pairs = n_atoms(H2O) * n_atoms(H2O2) = 3*4 = 12
    # number of pairs of pairs = n_pairs over 2 = 12 over 2 = 66
    # Number of bonds: 3 (H2O2) + 2 (H2O) = 5
    # Number of intramolecular unbound pairs: 3 (H2O2) + 1 (H2O) = 4
    # Expected number of hits:
    # Two intermol coordinates: 66
    # One intermol coordinate: 12
    # One intermol + one diss: 12*5 = 60
    # One intermol + one intramol formation = 12*4 = 48
    # Sum: 186
    n_trials = trial_generator.estimate_n_bimolecular_trials(structure_file1, structure_file2, attack_points_per_site=2)
    assert n_trials == 186

    # Influence of rotamers
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers = 2
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers_two_on_two = 3
    # Expected number of trials
    # Two intermol coordinates*number_rotamers_two_on_two: 66*3
    # One intermol coordinate*number_rotamers: 12*2
    # One intermol + one diss*number_rotamers : 12*5*2
    # One intermol + one intramol formation*number_rotamers = 12*4*2
    # Sum:
    n_trials = trial_generator.estimate_n_bimolecular_trials(structure_file1, structure_file2)
    assert n_trials == 438

    # Influence of multiple attack points
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers = 1
    trial_generator.options.bimolecular.complex_generator.options.number_rotamers_two_on_two = 1
    trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = True
    # Expected number of trials
    # 186 * 2**2
    # Sum: 744
    n_trials = trial_generator.estimate_n_bimolecular_trials(structure_file1, structure_file2, attack_points_per_site=2)
    assert n_trials == 744

    # Test with limited number of reactive pairs
    # Only one reactive intermolecular pair
    trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = False
    n_trials = trial_generator.estimate_n_bimolecular_trials(
        structure_file1, structure_file2, attack_points_per_site=2, n_inter_reactive_pairs=1
    )
    # Expected number of trials:
    # One intermol coordinate: 1
    # Two intermol coordinates: 0
    # One intermol + one diss: 1*5 = 5
    # One intermol + one intramol formation = 1*4 = 4
    # Sum: 10
    assert n_trials == 10

    # Only one reactive intermolecular pair
    trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = False
    n_trials = trial_generator.estimate_n_bimolecular_trials(
        structure_file1, structure_file2, attack_points_per_site=2, n_inter_reactive_pairs=1
    )
    # Expected number of trials:
    # One intermol coordinate: 1
    # Two intermol coordinates: 0
    # One intermol + one diss: 1*5 = 5
    # One intermol + one intramol formation = 1*4 = 4
    # Sum: 10
    assert n_trials == 10

    # Only two reactive diss pairs in H2O2
    trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = False
    n_trials = trial_generator.estimate_n_bimolecular_trials(
        structure_file1, structure_file2, attack_points_per_site=2, n_inter_reactive_pairs=1, n_reactive_bound_pairs1=2
    )
    # Expected number of trials:
    # One intermol coordinate: 1
    # Two intermol coordinates: 0
    # One intermol + one diss: 1*4 = 4
    # One intermol + one intramol formation = 1*4 = 4
    # Sum: 9
    assert n_trials == 9

    # No intramolecular reactive pairs in H2O
    trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = False
    n_trials = trial_generator.estimate_n_bimolecular_trials(
        structure_file1,
        structure_file2,
        attack_points_per_site=2,
        n_inter_reactive_pairs=1,
        n_reactive_bound_pairs1=2,
        n_reactive_unbound_pairs2=0,
    )
    # Expected number of trials:
    # One intermol coordinate: 1
    # Two intermol coordinates: 0
    # One intermol + one diss: 1*4 = 4
    # One intermol + one intramol formation = 1*3 = 3
    # Sum: 8
    assert n_trials == 8

    # No reactive intermolecular pairs
    trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = False
    n_trials = trial_generator.estimate_n_bimolecular_trials(
        structure_file1, structure_file2, attack_points_per_site=2, n_inter_reactive_pairs=0
    )
    assert n_trials == 0
