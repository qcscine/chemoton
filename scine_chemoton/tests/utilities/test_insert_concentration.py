#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
import random

# Third party imports
import scine_database as db

# Local application tests imports
from .. import test_database_setup as db_setup
from ...utilities.insert_concentration import insert_concentration_for_compound
from ...gears.kinetic_modeling.concentration_query_functions import (
    query_concentration_with_model_id,
    query_concentration_with_object,
    query_reaction_flux_with_model,
    query_reaction_fluxes_with_model,
    query_reaction_fluxes,
)


def test_insert_concentration_for_compound():
    n_compounds = 4
    n_flasks = 0
    n_reactions = 3
    max_r_per_c = 10
    max_n_products_per_r = 2
    max_n_educts_per_r = 1
    max_s_per_c = 1
    max_steps_per_r = 1
    barrier_limits = (10, 20)
    n_inserts = 3
    manager = db_setup.get_random_db(
        n_compounds,
        n_flasks,
        n_reactions,
        max_r_per_c,
        "chemoton_test_insert_concentration_for_compound",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )
    model = db.Model("FAKE", "FAKE", "F-AKE")
    compounds = manager.get_collection("compounds")
    flasks = manager.get_collection("flasks")
    properties = manager.get_collection("properties")
    structures = manager.get_collection("structures")
    reactions = manager.get_collection("reactions")
    compound = compounds.find(dumps({}))
    label = "start_concentration"

    model_two = db.Model("SOME_OTHER_FAKE", "", "")
    insert_concentration_for_compound(manager, 0.1, model, compound.id(), True, label)
    centroid = compound.get_centroid(manager)
    assert centroid.has_property(label)

    insert_concentration_for_compound(manager, 0.2, model, compound.id(), True, label)
    concentration = query_concentration_with_model_id(label, compound.id(), compounds, properties, structures, model)
    assert concentration == 0.2

    insert_concentration_for_compound(manager, 0.3, model, compound.id(), False, label)
    concentration = query_concentration_with_model_id(label, compound.id(), compounds, properties, structures, model)
    assert concentration == 0.3

    insert_concentration_for_compound(manager, 1.0, model_two, compound.id(), False, label)
    concentration = query_concentration_with_model_id(label, compound.id(), compounds, properties, structures, model)
    assert concentration == 0.3

    concentration = query_concentration_with_object(label, compound, properties, structures)
    assert concentration == 1.0

    reaction = reactions.get_one_reaction(dumps({}))
    reaction.link(reactions)
    value = 700.0
    flux_label = "_concentration_flux"
    insert_concentration_for_compound(manager, value, model, reaction.get_reactants(db.Side.LHS)[0][0], True,
                                      reaction.id().string() + flux_label)
    assert (query_reaction_flux_with_model(flux_label, reaction, compounds, flasks, structures, properties, model)
            - value) < 1e-9

    other_value = random.random()
    insert_concentration_for_compound(manager, other_value, model_two, reaction.get_reactants(db.Side.LHS)[0][0], False,
                                      reaction.id().string() + flux_label)

    random_values = random.sample(range(10, 30), 5)
    for val in random_values:
        insert_concentration_for_compound(manager, val, model, reaction.get_reactants(db.Side.LHS)[0][0], False,
                                          reaction.id().string() + flux_label)
    all_fluxes = query_reaction_fluxes(flux_label, reaction, compounds, flasks, structures, properties)
    assert len(all_fluxes) == len([value, other_value] + random_values)
    assert max(all_fluxes) == value
    for ref, result in zip([value, other_value] + random_values, all_fluxes):
        assert abs(ref - result) < 1e-9
    all_model_fluxes = query_reaction_fluxes_with_model(flux_label, reaction, compounds, flasks, structures, properties,
                                                        model)
    assert len(all_model_fluxes) == len([value] + random_values)
    for ref, result in zip([value] + random_values, all_model_fluxes):
        assert abs(ref - result) < 1e-9

    all_model_fluxes_two = query_reaction_fluxes_with_model(flux_label, reaction, compounds, flasks, structures,
                                                            properties, model_two)
    assert len(all_model_fluxes_two) == 1
    assert abs(all_model_fluxes_two[0] - other_value) < 1e-9

    # Cleaning
    manager.wipe()
