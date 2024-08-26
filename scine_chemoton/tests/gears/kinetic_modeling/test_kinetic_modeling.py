#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
from typing import List, Optional

# Third party imports
import scine_database as db
import scine_utilities as utils
from scine_database import test_database_setup as db_setup
from scine_database.insert_concentration import insert_concentration_for_compound

# Local application tests imports
from ....engine import Engine
from ....gears.kinetic_modeling.kinetic_modeling import KineticModeling
from ...utilities.db_object_wrappers.test_thermodynamic_properties import TestThermodynamicProperties
from ....utilities.model_combinations import ModelCombination
from ....utilities.db_object_wrappers.wrapper_caches import MultiModelCacheFactory
from ....gears.kinetic_modeling.rms_network_extractor import ReactionNetworkData


def test_random_kinetic_model():
    n_compounds = 10
    n_flasks = 3
    n_reactions = 10
    max_r_per_c = 10
    max_n_products_per_r = 2
    max_n_educts_per_r = 2
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (10, 20)
    n_inserts = 3
    MultiModelCacheFactory().clear()
    manager = db_setup.get_random_db(
        n_compounds,
        n_flasks,
        n_reactions,
        max_r_per_c,
        "chemoton_test_random_kinetic_model",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    model = db.Model("FAKE", "FAKE", "F-AKE")
    structures = manager.get_collection("structures")
    properties = manager.get_collection("properties")
    calculations = manager.get_collection("calculations")
    compounds = manager.get_collection("compounds")
    steps = manager.get_collection("elementary_steps")
    reactions = manager.get_collection("reactions")
    flasks = manager.get_collection("flasks")

    n_initial_reactions = reactions.count(dumps({}))

    # Add three more reactions. Two barrierless and one regular one.
    # lhs_c_id -> lhs_f_id -> TS -> rhs_f_id -> rhs_c_id
    random_compounds = compounds.random_select_compounds(4)
    lhs_c_id_one = random_compounds[0].id()
    lhs_c_id_two = random_compounds[1].id()
    rhs_c_id_one = random_compounds[2].id()
    rhs_c_id_two = random_compounds[3].id()
    lhs_compound_one = db.Compound(lhs_c_id_one, compounds)
    lhs_compound_two = db.Compound(lhs_c_id_two, compounds)
    rhs_compound_one = db.Compound(rhs_c_id_one, compounds)
    rhs_compound_two = db.Compound(rhs_c_id_two, compounds)

    lhs_f_id, lhs_comp_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_OPTIMIZED)
    rhs_f_id, rhs_comp_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_OPTIMIZED)
    no_e_f_id, no_e_comp_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_OPTIMIZED)
    lhs_complex = db.Structure(lhs_comp_id, structures)
    rhs_complex = db.Structure(rhs_comp_id, structures)
    db_setup.add_random_energy(lhs_complex, (70.0, 71.0), properties)
    db_setup.add_random_energy(rhs_complex, (70.0, 71.0), properties)

    lhs_s_id_one = lhs_compound_one.get_centroid()
    lhs_s_id_two = lhs_compound_two.get_centroid()
    rhs_s_id_one = rhs_compound_one.get_centroid()
    rhs_s_id_two = rhs_compound_two.get_centroid()
    no_e_flask = db.Flask(no_e_f_id, flasks)
    for compound in compounds.iterate_all_compounds():
        compound.link(compounds)
        compound.enable_exploration()
    for flask in flasks.iterate_all_flasks():
        flask.link(flasks)
        flask.enable_exploration()

    # set up steps between aggregates
    step_barrierless_lhs = db.ElementaryStep()
    step_barrierless_lhs.link(steps)
    step_barrierless_lhs.create([lhs_s_id_one, lhs_s_id_two], [lhs_comp_id])
    step_barrierless_lhs.set_type(db.ElementaryStepType.BARRIERLESS)

    step_barrierless_rhs = db.ElementaryStep()
    step_barrierless_rhs.link(steps)
    step_barrierless_rhs.create([rhs_comp_id], [rhs_s_id_one, rhs_s_id_two])
    step_barrierless_rhs.set_type(db.ElementaryStepType.BARRIERLESS)

    step_central = db.ElementaryStep()
    step_central.link(steps)
    step_central.create([lhs_comp_id], [rhs_comp_id])

    step_barrierless_incomplete = db.ElementaryStep()
    step_barrierless_incomplete.link(steps)
    step_barrierless_incomplete.create([no_e_comp_id], [rhs_s_id_one, rhs_s_id_two])
    step_barrierless_incomplete.set_type(db.ElementaryStepType.BARRIERLESS)

    # set up TS and energies
    lhs_comp_structure = db.Structure(lhs_comp_id, structures)
    db_setup.add_random_energy(lhs_comp_structure, (0.0, 1.0), properties)
    ts = db.Structure(db_setup.insert_single_empty_structure_aggregate(manager, db.Label.TS_GUESS)[1], structures)
    db_setup.add_random_energy(ts, (70.0, 71.0), properties)
    step_central.set_transition_state(ts.get_id())

    # set up reactions
    reaction_barrierless_lhs = db.Reaction()
    reaction_barrierless_lhs.link(reactions)
    reaction_barrierless_lhs.create([lhs_c_id_one, lhs_c_id_two], [lhs_f_id],
                                    [db.CompoundOrFlask.COMPOUND, db.CompoundOrFlask.COMPOUND],
                                    [db.CompoundOrFlask.FLASK])
    reaction_barrierless_lhs.set_elementary_steps([step_barrierless_lhs.get_id()])
    step_barrierless_lhs.set_reaction(reaction_barrierless_lhs.id())

    reaction_barrierless_rhs = db.Reaction()
    reaction_barrierless_rhs.link(reactions)
    reaction_barrierless_rhs.create([rhs_f_id], [rhs_c_id_one, rhs_c_id_two], [db.CompoundOrFlask.FLASK],
                                    [db.CompoundOrFlask.COMPOUND, db.CompoundOrFlask.COMPOUND])
    reaction_barrierless_rhs.set_elementary_steps([step_barrierless_rhs.get_id()])
    step_barrierless_rhs.set_reaction(reaction_barrierless_rhs.id())

    reaction_central = db.Reaction()
    reaction_central.link(reactions)
    reaction_central.create([lhs_f_id], [rhs_f_id], [db.CompoundOrFlask.FLASK], [db.CompoundOrFlask.FLASK])
    reaction_central.set_elementary_steps([step_central.get_id()])
    step_central.set_reaction(reaction_central.id())

    reaction_incomplete = db.Reaction()
    reaction_incomplete.link(reactions)
    reaction_incomplete.create([no_e_f_id], [rhs_c_id_one, rhs_c_id_two], [db.CompoundOrFlask.FLASK],
                               [db.CompoundOrFlask.COMPOUND, db.CompoundOrFlask.COMPOUND])
    reaction_incomplete.set_elementary_steps([step_barrierless_incomplete.get_id()])
    step_barrierless_incomplete.set_reaction(reaction_incomplete.id())

    lhs_flask = db.Flask(lhs_f_id, flasks)
    rhs_flask = db.Flask(rhs_f_id, flasks)
    lhs_compound_one.add_reaction(reaction_barrierless_lhs.id())
    lhs_compound_two.add_reaction(reaction_barrierless_lhs.id())
    lhs_flask.add_reaction(reaction_barrierless_lhs.id())
    lhs_flask.add_reaction(reaction_central.id())
    rhs_flask.add_reaction(reaction_central.id())
    rhs_flask.add_reaction(reaction_barrierless_rhs.id())
    rhs_compound_one.add_reaction(reaction_barrierless_rhs.id())
    rhs_compound_one.add_reaction(reaction_incomplete.id())
    rhs_compound_two.add_reaction(reaction_barrierless_rhs.id())
    rhs_compound_two.add_reaction(reaction_incomplete.id())
    no_e_flask.add_reaction(reaction_incomplete.id())

    # Set the starting concentrations for all compounds.
    for structure in structures.iterate_all_structures():
        structure.link(structures)
        con_prop = db.NumberProperty.make("start_concentration", model, 0.1, properties)
        con_prop.set_structure(structure.id())
        structure.add_property("start_concentration", con_prop.id())

    # Check the job-set up.
    kinetic_modeling_gear = KineticModeling()
    kinetic_modeling_gear.options.model_combinations = [ModelCombination(model, model)]
    kinetic_modeling_gear.options.job = db.Job("kinetx_kinetic_modeling")
    kinetic_modeling_gear.options.job_settings = KineticModeling.get_default_settings(kinetic_modeling_gear.options.job)
    kinetic_modeling_gear.options.cycle_time = 0.1
    kinetic_modeling_gear.options.max_barrier = float("inf")
    kinetic_modeling_gear.options.only_electronic = True

    kinetic_modeling_engine = Engine(manager.get_credentials(), fork=False)
    kinetic_modeling_engine.set_gear(kinetic_modeling_gear)
    kinetic_modeling_engine.run(single=True)
    assert calculations.count(dumps({})) == 1
    kinetic_modeling_engine.run(single=True)
    assert calculations.count(dumps({})) == 1

    # Test whether the gear notices that the same job is already in the database.
    gear2 = KineticModeling()
    gear2.options.job_settings = KineticModeling.get_default_settings(kinetic_modeling_gear.options.job)
    gear2.options.model_combinations = [ModelCombination(model, model)]
    gear2.options.cycle_time = 0.1
    gear2.options.only_electronic = True

    kinetic_modeling_engine2 = Engine(manager.get_credentials(), fork=False)
    kinetic_modeling_engine2.set_gear(gear2)
    kinetic_modeling_engine2.run(single=True)
    assert calculations.count(dumps({})) == 1

    for structure in structures.iterate_all_structures():
        structure.link(structures)
        con_prop = db.NumberProperty.make("concentration_flux", model, 0.1, properties)
        con_prop.set_structure(structure.id())
        structure.add_property("concentration_flux", con_prop.id())

    # Change the settings and test again.
    old_calc = calculations.find(dumps({}))
    old_calc.set_status(db.Status.COMPLETE)
    kinetic_modeling_engine2.run(single=True)
    assert calculations.count(dumps({})) == 1

    old_calculation_settings = old_calc.get_settings()
    assert len(old_calculation_settings["reaction_ids"]) <= n_initial_reactions + 3  # one reaction should be incomplete
    assert reactions.count(dumps({})) == n_initial_reactions + 4

    gear3 = KineticModeling()
    gear3.options.job_settings = KineticModeling.get_default_settings(kinetic_modeling_gear.options.job)
    gear3.options.job_settings["t_max"] = 1e+5
    gear3.options.model_combinations = [ModelCombination(model, model)]
    gear3.options.cycle_time = 0.1
    gear3.options.only_electronic = True

    kinetic_modeling_engine3 = Engine(manager.get_credentials(), fork=False)
    kinetic_modeling_engine3.set_gear(gear3)
    kinetic_modeling_engine3.run(single=True)
    assert calculations.count(dumps({})) == 2

    kinetic_modeling_engine3.run(single=True)
    kinetic_modeling_engine3.run(single=True)
    assert calculations.count(dumps({})) == 2

    old_calc = calculations.find(dumps({"status": "hold"}))
    old_calc.set_status(db.Status.COMPLETE)
    gear4 = KineticModeling()
    gear4.options.job = db.Job("rms_kinetic_modeling")
    gear4.options.job_settings = KineticModeling.get_default_settings(gear4.options.job)
    gear4.options.model_combinations = [ModelCombination(model, model)]
    gear4.options.cycle_time = 0.1
    gear4.options.only_electronic = True

    kinetic_modeling_engine4 = Engine(manager.get_credentials(), fork=False)
    kinetic_modeling_engine4.set_gear(gear4)
    kinetic_modeling_engine4.run(single=True)
    assert calculations.count(dumps({})) == 3
    old_calc = calculations.find(dumps({"status": "hold"}))
    old_calc.set_status(db.Status.COMPLETE)

    # disable one reaction
    reaction_barrierless_lhs.disable_exploration()
    reaction_barrierless_lhs.disable_analysis()
    kinetic_modeling_engine4.run(single=True)
    old_calc = calculations.find(dumps({"status": "hold"}))
    old_calculation_settings = old_calc.get_settings()
    assert len(old_calculation_settings["reaction_ids"]) <= n_initial_reactions + 2
    assert reaction_barrierless_lhs.id().string() not in old_calculation_settings["reaction_ids"]
    assert calculations.count(dumps({})) == 4
    old_calc.set_status(db.Status.COMPLETE)

    # disable all reactions
    for reaction in reactions.iterate_all_reactions():
        reaction.link(reactions)
        reaction.disable_analysis()
        reaction.disable_exploration()
    kinetic_modeling_engine4.run(single=True)
    assert calculations.count(dumps({})) == 4  # no reaction left. Therefore, there cannot be a new calculation.

    # disable one aggregate
    for reaction in reactions.iterate_all_reactions():
        reaction.link(reactions)
        reaction.enable_exploration()
        reaction.enable_analysis()
    lhs_flask.disable_exploration()
    lhs_flask.disable_analysis()
    kinetic_modeling_engine4.run(single=True)
    assert calculations.count(dumps({})) == 5
    old_calc = calculations.find(dumps({"status": "hold"}))
    old_calculation_settings = old_calc.get_settings()
    assert lhs_flask.id().string() not in old_calculation_settings["aggregate_ids"]
    old_calc.set_status(db.Status.COMPLETE)

    # disable all aggregates
    for compound in compounds.iterate_all_compounds():
        compound.link(compounds)
        compound.disable_exploration()
        compound.disable_analysis()
    for flask in flasks.iterate_all_flasks():
        flask.link(flasks)
        flask.disable_exploration()
        flask.disable_analysis()
    kinetic_modeling_engine4.run(single=True)
    assert calculations.count(dumps({})) == 5  # no aggregates left. Therefore, there cannot be a new calculation.

    # Cleaning
    manager.wipe()


def add_reaction(manager: db.Manager, ts_energy: float, lhs_energies: Optional[List[float]] = None,
                 rhs_energies: Optional[List[float]] = None, lhs_aggregate_ids: Optional[List[db.ID]] = None,
                 rhs_aggregate_ids: Optional[List[db.ID]] = None, model: Optional[db.Model] = None):
    structures = manager.get_collection("structures")
    properties = manager.get_collection("properties")
    compounds = manager.get_collection("compounds")
    if model is None:
        model = db_setup.get_fake_model()
    if lhs_aggregate_ids is None:
        lhs_aggregate_ids = []
        assert lhs_energies
        for e in lhs_energies:
            s, _ = TestThermodynamicProperties.get_OH_structures(model, e, structures, properties)
            hessian_property = TestThermodynamicProperties.get_OH_hessian(model, properties)
            s.add_property("hessian", hessian_property.id())
            hessian_property.set_structure(s.id())
            c = db.Compound()
            c.link(compounds)
            c.create([s.id()])
            lhs_aggregate_ids.append(c.id())
    if rhs_aggregate_ids is None:
        rhs_aggregate_ids = []
        assert rhs_energies
        for e in rhs_energies:
            s, _ = TestThermodynamicProperties.get_OH_structures(model, e, structures, properties)
            hessian_property = TestThermodynamicProperties.get_OH_hessian(model, properties)
            s.add_property("hessian", hessian_property.id())
            hessian_property.set_structure(s.id())
            c = db.Compound()
            c.link(compounds)
            c.create([s.id()])
            rhs_aggregate_ids.append(c.id())
    lhs_aggregates = [db.Compound(c_id, compounds) for c_id in lhs_aggregate_ids]
    rhs_aggregates = [db.Compound(c_id, compounds) for c_id in rhs_aggregate_ids]
    lhs_structure_ids = [c.get_centroid() for c in lhs_aggregates]
    rhs_structure_ids = [c.get_centroid() for c in rhs_aggregates]

    elementary_steps = manager.get_collection("elementary_steps")
    reactions = manager.get_collection("reactions")
    ts, _ = TestThermodynamicProperties.get_OH_structures(model, ts_energy, structures, properties)
    ts.set_label(db.Label.TS_OPTIMIZED)
    hessian_property = TestThermodynamicProperties.get_OH_hessian(model, properties)
    ts.add_property("hessian", hessian_property.id())
    hessian_property.set_structure(ts.id())
    new_step = db.ElementaryStep()
    new_step.link(elementary_steps)
    new_step.create(lhs_structure_ids, rhs_structure_ids)
    new_step.set_transition_state(ts.id())
    new_reaction = db.Reaction()
    new_reaction.link(reactions)
    new_reaction.create(lhs_aggregate_ids, rhs_aggregate_ids, [db.CompoundOrFlask.COMPOUND for _ in lhs_aggregate_ids],
                        [db.CompoundOrFlask.COMPOUND for _ in rhs_aggregate_ids])
    new_reaction.add_elementary_step(new_step.id())
    new_step.set_reaction(new_reaction.id())
    for c in lhs_aggregates + rhs_aggregates:
        c.add_reaction(new_reaction.id())

    return new_reaction, lhs_aggregates, rhs_aggregates


def test_single_reaction():
    MultiModelCacheFactory().clear()
    manager = db_setup.get_clean_db("chemoton_test_kinetic_modeling_single_reaction")
    compounds = manager.get_collection("compounds")
    calculations = manager.get_collection("calculations")
    model = db_setup.get_fake_model()

    hess_contribution = 28315.14304
    e1 = -634.6730820353
    e1_jpermol = e1 * utils.KJPERMOL_PER_HARTREE * 1e+3
    e2 = -634.6568134574
    e2_jpermol = e2 * utils.KJPERMOL_PER_HARTREE * 1e+3
    ets = -634.6145146309
    barrier_eh = ets - e1
    barrier = barrier_eh * utils.KJPERMOL_PER_HARTREE * 1e+3
    low_barrier = 0.9 * barrier
    low_barrier_2 = 0.8 * barrier
    _, lhs_a1, rhs_a1 = add_reaction(manager, ets, [e1], [e2])
    insert_concentration_for_compound(manager, 0.1, model, lhs_a1[0].id(), False, "start_concentration")

    gear = KineticModeling()
    gear.options.job = db.Job("rms_kinetic_modeling")
    gear.options.job_settings = KineticModeling.get_default_settings(gear.options.job)
    gear.options.model_combinations = [ModelCombination(model, model)]
    gear.options.cycle_time = 0.1
    kinetic_modeling_engine = Engine(manager.get_credentials(), fork=False)
    kinetic_modeling_engine.set_gear(gear)
    kinetic_modeling_engine.run(single=True)
    assert calculations.count(dumps({})) == 1
    kinetic_modeling_engine.run(single=True)
    assert calculations.count(dumps({})) == 1

    calc = calculations.find(dumps({"status": "hold"}))
    calc_settings = calc.get_settings()
    assert len(calc_settings["ea"]) == 1
    assert abs(calc_settings["ea"][0] - barrier) < 1e-2
    assert len(calc_settings["enthalpies"]) == 2
    assert abs(calc_settings["enthalpies"][0] - e1_jpermol - hess_contribution) < 1e-2
    assert abs(calc_settings["enthalpies"][1] - e2_jpermol - hess_contribution) < 1e-2

    calc.set_status(db.Status.COMPLETE)
    for c in compounds.iterate_all_compounds():
        c.link(compounds)
        insert_concentration_for_compound(manager, 0.1, model, c.id(), False, "concentration_flux")

    _, _, _ = add_reaction(manager, ets, [e1], [e2])  # add  disconnected reaction.
    kinetic_modeling_engine.run(single=True)  # this should not add a new calculation (already set up)
    assert calculations.count(dumps({})) == 1
    assert len(calc_settings["ea"]) == 1
    assert abs(calc_settings["ea"][0] - barrier) < 1e-2  # barrier in J/mol, so 1e-2 is approx 1e-8 E_h

    _, _, _ = add_reaction(manager, ets + 2 * barrier_eh, None, [e2],
                           lhs_aggregate_ids=[lhs_a1[0].id()])  # add  reaction with high barrier.
    gear.options.max_barrier = 2 * barrier * 1e-3
    kinetic_modeling_engine.run(single=True)  # this should not add a new calculation (already set up)
    assert calculations.count(dumps({})) == 1

    _, _, _ = add_reaction(manager, ets - 0.1 * barrier_eh, None, [e2],
                           lhs_aggregate_ids=[lhs_a1[0].id()])  # add  reaction with low barrier.
    kinetic_modeling_engine.run(single=True)  # this should add a new calculation with two reactions
    assert calculations.count(dumps({})) == 2
    calc = calculations.find(dumps({"status": "hold"}))
    calc_settings = calc.get_settings()
    assert len(calc_settings["ea"]) == 2
    calc.set_status(db.Status.COMPLETE)
    for c in compounds.iterate_all_compounds():
        c.link(compounds)
        insert_concentration_for_compound(manager, 0.1, model, c.id(), False, "concentration_flux")

    insert_concentration_for_compound(manager, 0.0, model, rhs_a1[0].id(), True, "concentration_flux")
    kinetic_modeling_engine.run(single=True)  # this should add a new calculation with one reaction and the low barrier
    assert calculations.count(dumps({})) == 3
    calc = calculations.find(dumps({"status": "hold"}))
    calc_settings = calc.get_settings()
    assert len(calc_settings["ea"]) == 1
    assert abs(calc_settings["ea"][0] - low_barrier) < 1e-3
    calc.set_status(db.Status.COMPLETE)

    only_barriers_model = db.Model("Only", "barriers", "model")
    _, _, _ = add_reaction(manager, ets - 0.2 * barrier_eh, None, [e2],
                           lhs_aggregate_ids=[lhs_a1[0].id()], rhs_aggregate_ids=[rhs_a1[0].id()],
                           model=only_barriers_model)
    structures = manager.get_collection("structures")
    properties = manager.get_collection("properties")
    for c, e in zip([lhs_a1[0], rhs_a1[0]], [e1, e2]):
        s, _ = TestThermodynamicProperties.get_OH_structures(only_barriers_model, e, structures, properties)
        hessian_property = TestThermodynamicProperties.get_OH_hessian(only_barriers_model, properties)
        s.add_property("hessian", hessian_property.id())
        hessian_property.set_structure(s.id())
        c.add_structure(s.id())

    gear2 = KineticModeling()
    gear2.options.job = db.Job("rms_kinetic_modeling")
    gear2.options.job_settings = KineticModeling.get_default_settings(gear.options.job)
    gear2.options.model_combinations = [ModelCombination(model)]
    gear2.options.model_combinations_reactions = [ModelCombination(only_barriers_model)]
    gear2.options.cycle_time = 0.1
    kinetic_modeling_engine2 = Engine(manager.get_credentials(), fork=False)
    kinetic_modeling_engine2.set_gear(gear2)
    kinetic_modeling_engine2.run(single=True)
    assert calculations.count(dumps({})) == 4

    calc = calculations.find(dumps({"status": "hold"}))
    calc_settings = calc.get_settings()
    assert len(calc_settings["ea"]) == 1
    assert abs(calc_settings["ea"][0] - low_barrier_2) < 1e-2
    assert len(calc_settings["enthalpies"]) == 2
    assert abs(calc_settings["enthalpies"][0] - e1_jpermol - hess_contribution) < 1e-2
    assert abs(calc_settings["enthalpies"][1] - e2_jpermol - hess_contribution) < 1e-2
    calc.set_status(db.Status.COMPLETE)

    for c in compounds.iterate_all_compounds():
        c.link(compounds)
        insert_concentration_for_compound(manager, 0.1, model, c.id(), False, "concentration_flux")
    gear2.options.model_combinations_reactions = [ModelCombination(only_barriers_model), ModelCombination(model)]
    gear2.reset_job_factory()
    kinetic_modeling_engine2.set_gear(gear2)
    kinetic_modeling_engine2.run(single=True)
    assert calculations.count(dumps({})) == 5

    calc = calculations.find(dumps({"status": "hold"}))
    calc_settings = calc.get_settings()
    assert len(calc_settings["ea"]) == 3
    assert abs(calc_settings["ea"][0] - barrier) < 1e-2 or abs(calc_settings["ea"][1] - barrier) < 1e-2 or abs(
        calc_settings["ea"][2] - barrier) < 1e-2
    assert abs(calc_settings["ea"][0] - low_barrier_2) < 1e-2 or abs(calc_settings["ea"][1] - low_barrier_2) < 1e-2\
        or abs(calc_settings["ea"][2] - low_barrier_2) < 1e-2
    assert abs(calc_settings["ea"][0] - low_barrier) < 1e-2 or abs(calc_settings["ea"][1] - low_barrier) < 1e-2 or\
        abs(calc_settings["ea"][2] - low_barrier) < 1e-2
    assert len(calc_settings["enthalpies"]) == 3
    assert abs(calc_settings["enthalpies"][0] - e1_jpermol - hess_contribution) < 1e-2 or abs(
        calc_settings["enthalpies"][1] - e1_jpermol - hess_contribution) < 1e-2 or abs(
        calc_settings["enthalpies"][2] - e1_jpermol - hess_contribution) < 1e-2

    # Test if the same data is provided by the explicit network extraction.
    data = ReactionNetworkData(manager, gear2.options, gear2.options.energy_references)
    assert len(data.ea) == 3
    assert abs(data.ea[0] - barrier) < 1e-2 or abs(data.ea[1] - barrier) < 1e-2 or abs(data.ea[2] - barrier) < 1e-2
    assert abs(data.ea[0] - low_barrier_2) < 1e-2 or abs(data.ea[1] - low_barrier_2) < 1e-2 or abs(
        data.ea[2] - low_barrier_2) < 1e-2
    assert abs(data.ea[0] - low_barrier) < 1e-2 or abs(data.ea[1] - low_barrier) < 1e-2 or abs(
        data.ea[2] - low_barrier) < 1e-2
    assert len(data.enthalpies) == 3
    assert abs(data.enthalpies[0] - e1_jpermol - hess_contribution) < 1e-2 or abs(
        data.enthalpies[1] - e1_jpermol - hess_contribution) < 1e-2 or abs(
        data.enthalpies[2] - e1_jpermol - hess_contribution) < 1e-2
    assert len(data.entropies) == 3

    # Cleaning
    manager.wipe()
