#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
import json
import pytest

# Third party imports
import scine_database as db
from scine_database import test_database_setup as db_setup

# Local application tests imports
from ....resources import resources_root_path

# Local application imports
from .....gears.elementary_steps.trial_generator.template_based import TemplateBased


def test_bimol():
    """
    Tests whether bimolecular elementary step trials are set up correctly
    with the bond-based trial generator
    """
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_bimol_template_based")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")

    # Add fake data
    model = db.Model("FAKE", "FAKE", "F-AKE")
    rr = resources_root_path()
    structure_list = []
    for mol in ["substrate", "hydrogenperoxide"]:
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
    trial_generator = TemplateBased()
    trial_generator.options.model = model
    trial_generator.options.reaction_template_file = os.path.join(rr, "test.rtdb.pickle.obj")
    trial_generator.initialize_collections(manager)
    # Generate trials
    trial_generator.bimolecular_reactions(structure_list)

    # Checks
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 96

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 2
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_model() == model
        assert calculation.get_job().order == "scine_react_complex_nt2"
        calculation.wipe()
    for structure in structure_list:
        structure.clear_all_calculations()

    # Run a second time
    trial_generator.bimolecular_reactions(structure_list)

    # Check again
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 96

    # Run a third time without deleting
    trial_generator.bimolecular_reactions(structure_list)

    # Check again
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 96

    # Rerun with a different model
    model2 = db.Model("FAKE2", "", "")
    trial_generator.options.model = model2
    trial_generator.options.energy_cutoff = 300.0
    trial_generator.options.enforce_atom_shapes = False
    trial_generator.bimolecular_reactions(structure_list)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 192

    # Cleaning
    manager.wipe()


def test_unimol():
    """
    Tests whether unimolecular elementary step trials containing only
    dissociations are set up correctly with the bond-based trial generator
    """
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_unimol_dissociations_template_based")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")

    # Add fake data
    model = db.Model("FAKE", "FAKE", "F-AKE")
    rr = resources_root_path()
    for mol in ["substrate"]:
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
    structure.clear_all_calculations()

    # Setup trial generator with settings that cannot be fulfilled
    trial_generator = TemplateBased()
    trial_generator.initialize_collections(manager)
    trial_generator.options.reaction_template_file = os.path.join(rr, "test.rtdb.pickle.obj")
    trial_generator.options.model = model

    trial_generator.unimolecular_reactions(structure)
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 123

    # Run a second time
    trial_generator.unimolecular_reactions(structure)

    # Check again
    hits = structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    assert len(hits) == 0
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 123

    for hit in hits:
        calculation = db.Calculation(hit.id())
        calculation.link(calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_react_complex_nt2"
        assert calculation.get_model() == model

    # Cleaning
    manager.wipe()


def test_estimate_n_unimolecular_trials():
    """
    Tests whether the number of trial reactive coordinates forecasted for
    unimolecular reactions is correct
    """
    rr = resources_root_path()
    structure_file = os.path.join(rr, "proline.xyz")
    trial_generator = TemplateBased()
    trial_generator.options.reaction_template_file = os.path.join(rr, "test.rtdb.pickle.obj")

    with pytest.raises(NotImplementedError):
        n_trials = trial_generator.estimate_n_unimolecular_trials(structure_file)
        assert n_trials == 0

    with pytest.raises(NotImplementedError):
        trial_generator.options.energy_cutoff = 500
        n_trials = trial_generator.estimate_n_unimolecular_trials(structure_file)
        assert n_trials == 0


def test_estimate_n_bimolecular_trials():
    """
    Tests whether the number of trial reactive coordinates forecasted for
    unimolecular reactions is correct
    """
    rr = resources_root_path()
    structure_file1 = os.path.join(rr, "proline.xyz")
    structure_file2 = os.path.join(rr, "propanal.xyz")
    trial_generator = TemplateBased()
    trial_generator.options.reaction_template_file = os.path.join(rr, "test.rtdb.pickle.obj")

    with pytest.raises(NotImplementedError):
        n_trials = trial_generator.estimate_n_unimolecular_trials(structure_file1, structure_file2)
        assert n_trials == 0

    with pytest.raises(NotImplementedError):
        trial_generator.options.energy_cutoff = 500
        n_trials = trial_generator.estimate_n_unimolecular_trials(structure_file1, structure_file2)
        assert n_trials == 0
