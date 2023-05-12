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
from .....gears.elementary_steps.trial_generator.fast_dissociations import FastDissociations


def test_unimol_fast_dissociations():
    """
    Tests whether unimolecular elementary step trials containing only
    dissociations are set up correctly with the fast dissociations trial generator
    """
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_unimol_fast_dissociations")

    # Get collections
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")

    # Add fake data
    model = db.Model("FAKE", "", "")
    rr = resources_root_path()
    for mol in ["cyclohexene"]:
        graph = json.load(open(os.path.join(rr, mol + ".json"), "r"))
        structure = db.Structure(db.ID(), structures)
        structure.create(os.path.join(rr, mol + ".xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", graph["masm_idx_map"])
        structure.set_graph("masm_decision_list", graph["masm_decision_list"])
        compound = db.Compound(db.ID(), compounds)
        compound.create([structure.id()])
        structure.set_aggregate(compound.id())

    """ 0-0 """
    # Setup trial generator with settings that cannot be fulfilled
    trial_generator = FastDissociations()
    trial_generator.options.always_further_explore_dissociative_reactions = False
    trial_generator.options.enable_further_explorations = False
    trial_generator.initialize_collections(manager)
    trial_generator.options.model = model
    trial_generator.options.min_bond_dissociations = 0
    trial_generator.options.max_bond_dissociations = 0
    # Generate no trials
    trial_generator.unimolecular_reactions(structure)

    # Checks
    assert not structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 0
    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 0

    # with dissociative explorations
    trial_generator.options.always_further_explore_dissociative_reactions = True
    trial_generator.unimolecular_reactions(structure)
    assert not structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 0
    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 0

    # with further explorations
    trial_generator.options.enable_further_explorations = True
    trial_generator.unimolecular_reactions(structure)
    assert not structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 0
    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 0

    """ 1-1 """
    # Exactly one dissociation at once
    trial_generator.options.always_further_explore_dissociative_reactions = False
    trial_generator.options.enable_further_explorations = False
    trial_generator.options.min_bond_dissociations = 1
    trial_generator.options.max_bond_dissociations = 1
    # Generate trials
    trial_generator.unimolecular_reactions(structure)
    # Expected disconnections are the 10 C-H bonds
    assert not structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 10

    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    # Check again
    assert not structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 10

    # Run a third time with exact settings check
    trial_generator.unimolecular_reactions(structure, with_exact_settings_check=True)
    # Check again
    assert not structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 10

    # with dissociative explorations
    trial_generator.options.always_further_explore_dissociative_reactions = True
    trial_generator.unimolecular_reactions(structure)
    assert not structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    more_hits = calculations.query_calculations(json.dumps({}))
    # Expect previous 10 + 6 C-C bonds
    assert len(more_hits) == 16
    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    more_hits = calculations.query_calculations(json.dumps({}))
    assert len(more_hits) == 16

    # with further explorations
    trial_generator.options.enable_further_explorations = True
    trial_generator.unimolecular_reactions(structure)
    assert not structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    most_hits = calculations.query_calculations(json.dumps({}))
    assert len(most_hits) == len(hits) + len(more_hits)
    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    most_hits = calculations.query_calculations(json.dumps({}))
    assert len(most_hits) == len(hits) + len(more_hits)

    for hit in hits:
        calculation = db.Calculation(hit.id(), calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_dissociation_cut"
        assert calculation.get_model() == model
        n_diss = len(calculation.get_setting("dissociations"))
        assert n_diss == 2  # Two entries per pair

    for hit in more_hits:
        calculation = db.Calculation(hit.id(), calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        order = calculation.get_job().order
        assert order in ["scine_dissociation_cut", "scine_react_complex_nt2"]
        assert calculation.get_model() == model
        n_diss = calculation.get_settings().get("dissociations")
        n_diss_nt = calculation.get_settings().get("nt_nt_dissociations")
        assert n_diss is None and len(n_diss_nt) == 2 or len(n_diss) == 2 and n_diss_nt is None  # Two entries per pair
        elements = structure.get_atoms().elements
        if order == "scine_dissociation_cut":
            lhs, rhs = calculation.get_setting("dissociations")
            assert elements[lhs] != elements[rhs]
        else:
            lhs, rhs = calculation.get_setting("nt_nt_dissociations")
            assert elements[lhs] == elements[rhs]

    for hit in most_hits:
        calculation = db.Calculation(hit.id(), calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order in ["scine_dissociation_cut", "scine_react_complex_nt2"]
        assert calculation.get_model() == model
        n_diss = calculation.get_settings().get("dissociations")
        n_diss_nt = calculation.get_settings().get("nt_nt_dissociations")
        assert n_diss is None and len(n_diss_nt) == 2 or len(n_diss) == 2 and n_diss_nt is None  # Two entries per pair
        # Cleaning
        calculation.wipe()

    """ 2-2 """
    # Try exactly two dissociations at once
    trial_generator.options.always_further_explore_dissociative_reactions = False
    trial_generator.options.enable_further_explorations = False
    trial_generator.options.min_bond_dissociations = 2
    trial_generator.options.max_bond_dissociations = 2
    # Generate trials
    trial_generator.unimolecular_reactions(structure)

    # Expected number of hits: if you break 2, all reactions are dissconnective for cyclohexene
    # 16 bonds choose 2 = 10 choose 2 = 120
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 120
    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 120

    # with dissociative explorations
    trial_generator.options.always_further_explore_dissociative_reactions = True
    trial_generator.unimolecular_reactions(structure)
    assert not structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    more_hits = calculations.query_calculations(json.dumps({}))
    # No pure dissociatives
    assert len(more_hits) == len(hits)
    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    more_hits = calculations.query_calculations(json.dumps({}))
    assert len(more_hits) == len(hits)

    # with further explorations
    trial_generator.options.enable_further_explorations = True
    trial_generator.unimolecular_reactions(structure)
    assert not structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    most_hits = calculations.query_calculations(json.dumps({}))
    assert len(most_hits) == len(hits) + len(more_hits)
    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    most_hits = calculations.query_calculations(json.dumps({}))
    assert len(most_hits) == len(hits) + len(more_hits)

    for hit in hits:
        calculation = db.Calculation(hit.id(), calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_dissociation_cut"
        n_diss = len(calculation.get_setting("dissociations"))
        assert n_diss == 4

    for hit in more_hits:
        calculation = db.Calculation(hit.id(), calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order == "scine_dissociation_cut"
        n_diss = len(calculation.get_setting("dissociations"))
        assert n_diss == 4

    for hit in most_hits:
        calculation = db.Calculation(hit.id(), calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order in ["scine_dissociation_cut", "scine_react_complex_nt2"]
        assert calculation.get_model() == model
        n_diss = calculation.get_settings().get("dissociations")
        n_diss_nt = calculation.get_settings().get("nt_nt_dissociations")
        assert n_diss is None and len(n_diss_nt) == 4 or len(n_diss) == 4 and n_diss_nt is None

    """ 0-2 """
    trial_generator.options.always_further_explore_dissociative_reactions = False
    trial_generator.options.enable_further_explorations = False
    # Try one or two dissociations at once
    trial_generator.options.min_bond_dissociations = 0
    trial_generator.options.max_bond_dissociations = 2
    # Generate trials
    trial_generator.unimolecular_reactions(structure, with_exact_settings_check=True)

    # Expected number of hits
    # Single dissociations = 10 C-H bonds
    # Double dissociations = number of bonds choose 2 = 16 choose 2 = 120, but already done
    # Sum: 10 + 240 = 250
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 250
    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    hits = calculations.query_calculations(json.dumps({}))
    assert len(hits) == 250

    # with dissociative explorations
    trial_generator.options.always_further_explore_dissociative_reactions = True
    trial_generator.unimolecular_reactions(structure)
    assert not structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    more_hits = calculations.query_calculations(json.dumps({}))
    # 6 C-C pure single dissociations
    assert len(more_hits) == len(hits) + 6
    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    more_hits = calculations.query_calculations(json.dumps({}))
    assert len(more_hits) == len(hits) + 6

    # with further explorations
    trial_generator.options.enable_further_explorations = True
    trial_generator.unimolecular_reactions(structure)
    assert not structures.query_structures(json.dumps({"label": "reactive_complex_guess"}))
    most_hits = calculations.query_calculations(json.dumps({}))
    # 10 new fast + 6 pure single dissociations
    assert len(most_hits) == 250 + 10 + 6
    # Run a second time
    trial_generator.unimolecular_reactions(structure)
    most_hits = calculations.query_calculations(json.dumps({}))
    assert len(most_hits) == 250 + 10 + 6

    for hit in hits + more_hits + most_hits:
        calculation = db.Calculation(hit.id(), calculations)
        assert len(calculation.get_structures()) == 1
        assert calculation.get_status() == db.Status.HOLD
        assert calculation.get_job().order in ["scine_dissociation_cut", "scine_react_complex_nt2"]
        assert calculation.get_model() == model
        n_diss = calculation.get_settings().get("dissociations")
        n_diss_nt = calculation.get_settings().get("nt_nt_dissociations")
        assert n_diss is None and len(n_diss_nt) in (2, 4) or len(n_diss) in (2, 4) and n_diss_nt is None

    # Cleaning
    manager.wipe()
