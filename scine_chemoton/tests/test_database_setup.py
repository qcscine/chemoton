#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import List, Tuple
from json import dumps
import os
import random

# Third party imports
import scine_database as db
import scine_utilities as utils

# local imports
from .resources import resources_root_path
from ..utilities.queries import identical_reaction


def get_test_db_credentials(name: str = "chemoton_unittests") -> db.Credentials:
    """
    Generate a set of credentials pointing to a database and server.
    The server IP and port are assumed to be `127.0.0.1` and `27017`
    unless specified otherwise with the environment variables
    ``TEST_MONGO_DB_IP`` and ``TEST_MONGO_DB_PORT``.

    Parameters
    ----------
    name :: str
        The name of the database to connect to.

    Returns
    -------
    result :: db.Credentials
        The credentials to access the test database.
    """
    ip = os.environ.get('TEST_MONGO_DB_IP', "127.0.0.1")
    port = os.environ.get('TEST_MONGO_DB_PORT', "27017")
    return db.Credentials(ip, int(port), name)


def get_clean_db(name: str = "chemoton_unittests") -> db.Manager:
    """
    Generate a clean database using ``get_test_db_credentials`` to determine
    the database IP.

    Parameters
    ----------
    name :: str
        The name of the database to connect to.

    Returns
    -------
    result :: db.Manager
        The database manager, connected to the requested server and named
        database.
    """
    credentials = get_test_db_credentials(name)
    manager = db.Manager()
    manager.set_credentials(credentials)
    try:
        manager.connect()
        manager.wipe()
    except BaseException:
        manager.wipe(True)
        manager.connect()
    manager.init()
    return manager


def get_random_db(
    n_compounds: int,
    n_flasks: int,
    n_reactions: int,
    max_r_per_c: int,
    name: str = "chemoton_unittests",
    max_n_products_per_r: int = 4,
    max_n_educts_per_r: int = 2,
    max_s_per_c: int = 3,
    max_steps_per_r: int = 3,
    barrier_limits: Tuple[float, float] = (1.0, 1500.0),
    n_inserts: int = 2,
) -> db.Manager:
    """
    This generates a random network of a given number of reactions and compounds.

    Parameters
    ----------
    n_compounds :: int
        The number of compounds each with a random number of structures.
    n_flasks :: int
        The number of flasks each with a random number of structures.
    n_reactions :: int
        The number of reactions each with a random number of elementary steps.
    max_r_per_c :: int
        The maximum number of reactions per compound.
    name :: str
        The name of the database to connect to.
    max_n_products_per_r :: int
        The maximum number of possible products for a single reaction.
    max_n_educts_per_r :: int
        The maximum number of possible educts for a single reaction.
    max_s_per_c :: int
        The maximum number of possible structures for a single compound.
    max_steps_per_r :: int
        The maximum number of possible steps for a single reaction.
    barrier_limits :: Tuple[float, float]
        The lowest and highest possible barrier for a single reaction in kJ/mol.
    n_inserts :: int
        The number of compounds in the network that are marked as inserted by the user.

    Returns
    -------
    result :: db.Manager
        The database manager, connected to the requested server and named
        database.
    """
    assert max_n_products_per_r > 0
    assert max_n_educts_per_r > 0
    assert n_reactions > 0
    assert n_inserts > 0
    assert n_inserts <= n_compounds
    # enough reactions and allowed products to create the non-user-inserted compounds
    assert n_compounds < (n_reactions - n_flasks) * max_n_products_per_r
    # enough compounds for the biggest possible reaction
    assert n_compounds > max_n_products_per_r + max_n_educts_per_r
    # at least 2 compounds per reaction -> maximum allowed number of reactions
    assert 0.5 * n_compounds * max_r_per_c > n_reactions
    # worst case all reactions are biggest possible reaction
    assert n_compounds * max_r_per_c > n_reactions * (max_n_products_per_r + max_n_educts_per_r)

    manager = get_clean_db(name)
    properties = manager.get_collection("properties")
    compounds = manager.get_collection("compounds")
    flasks = manager.get_collection("flasks")
    reactions = manager.get_collection("reactions")
    structures = manager.get_collection("structures")
    elementary_steps = manager.get_collection("elementary_steps")

    for _ in range(n_inserts):
        _create_compound(max_s_per_c, properties, compounds, structures, user_input=True)

    for _ in range(n_reactions - n_flasks):
        _create_reaction(
            n_compounds,
            max_s_per_c,
            max_r_per_c,
            max_n_educts_per_r,
            max_n_products_per_r,
            max_steps_per_r,
            barrier_limits,
            reactions,
            compounds,
            properties,
            structures,
            elementary_steps,
        )

    # not guaranteed to have all necessary compounds yet
    # create missing ones and add them as products on reactions where we still have space for them
    n_missing_compounds = n_compounds - compounds.count("{}")
    for _ in range(n_missing_compounds):
        compound_id = _create_compound(max_s_per_c, properties, compounds, structures, user_input=False)
        compound = db.Compound(compound_id)
        compound.link(compounds)
        got_reaction = False
        # check all reactions for space left
        for reaction in reactions.iterate_all_reactions():
            reaction.link(reactions)
            products = reaction.get_reactants(db.Side.RHS)[1]
            if len(products) < max_n_products_per_r:
                products.append(compound_id)
                reaction.set_reactants(products, db.Side.RHS)
                compound.set_reactions([reaction.get_id()])
                _redo_elementary_steps(
                    reaction, max_steps_per_r, barrier_limits, compounds, structures, elementary_steps, properties
                )
                got_reaction = True
                break

        # if none left, substitute products of reactions that are produced with multiple reactions
        while not got_reaction:
            random_reaction = reactions.random_select_reactions(1)[0]
            random_reaction.link(reactions)
            products = random_reaction.get_reactants(db.Side.RHS)[1]
            for product_id in products:
                selection = {"rhs": {"$elemMatch": {"id": {"$oid": str(product_id)}}}}
                if reactions.count(dumps(selection)) > 1:
                    multiple_product = db.Compound(product_id)
                    multiple_product.link(compounds)
                    multiple_product.remove_reaction(random_reaction.get_id())
                    random_reaction.remove_reactant(product_id, db.Side.RHS)
                    random_reaction.add_reactant(compound.get_id(), db.Side.RHS, db.CompoundOrFlask.COMPOUND)
                    _redo_elementary_steps(
                        random_reaction,
                        max_steps_per_r,
                        barrier_limits,
                        compounds,
                        structures,
                        elementary_steps,
                        properties,
                    )
                    got_reaction = True
                    break

    possibilities = []
    for reaction in reactions.iterate_all_reactions():
        possibilities.append((reaction.get_id(), db.Side.LHS))
        possibilities.append((reaction.get_id(), db.Side.RHS))
    random.shuffle(possibilities)
    for lot in range(n_flasks):
        rid, side = possibilities[lot]
        _insert_flask(
            rid,
            side,
            flasks,
            structures,
            reactions,
            elementary_steps,
            properties,
            compounds
        )

    assert compounds.count("{}") == n_compounds
    assert reactions.count("{}") == n_reactions
    return manager


def _insert_flask(
    rid,
    side,
    flasks,
    structures,
    reactions,
    elementary_steps,
    properties,
    compounds
):
    # Generate complex and flask
    flask = db.Flask()
    flask.link(flasks)
    flask.create([], [])
    flask.disable_exploration()
    complex = db.Structure()
    complex.link(structures)
    complex.create(os.path.join(resources_root_path(), "water.xyz"), 0, 1)
    complex.set_label(db.Label.COMPLEX_OPTIMIZED)
    complex.set_aggregate(flask.get_id())
    complex.set_model(db.Model("FAKE", "FAKE", "F-AKE"))
    add_random_energy(complex, (-10000.0, -1000.0), properties)
    flask.add_structure(complex.get_id())
    # Update original reaction and elementary steps
    reaction = db.Reaction(rid)
    reaction.link(reactions)
    reactants = reaction.get_reactants(side)
    if side == db.Side.LHS:
        reactants = reactants[0]
    else:
        reactants = reactants[1]
    reaction.set_reactants([flask.get_id()], side, [db.CompoundOrFlask.FLASK])
    for esid in reaction.get_elementary_steps():
        es = db.ElementaryStep(esid)
        es.link(elementary_steps)
        step_reactants = es.get_reactants(side)
        if side == db.Side.LHS:
            step_reactants = step_reactants[0]
        else:
            step_reactants = step_reactants[1]
        for srid in step_reactants:
            s = db.Structure(srid)
            s.link(structures)
            assert s.get_compound() in reactants
        es.set_reactants([complex.get_id()], side)
    # Insert barrierless reaction with elementary step
    new_reaction = db.Reaction()
    new_reaction.link(reactions)
    new_step = db.ElementaryStep()
    new_step.link(elementary_steps)
    if side == db.Side.LHS:
        new_step.create([r for r in step_reactants], [complex.get_id()])
        new_reaction.create(
            reactants,
            [flask.get_id()],
            [db.CompoundOrFlask.COMPOUND for _ in reactants],
            [db.CompoundOrFlask.FLASK],
        )
    else:
        new_step.create([complex.get_id()], [r for r in step_reactants])
        new_reaction.create(
            [flask.get_id()],
            reactants,
            [db.CompoundOrFlask.FLASK],
            [db.CompoundOrFlask.COMPOUND for _ in reactants],
        )
    new_reaction.add_elementary_step(new_step.get_id())
    new_step.set_reaction(new_reaction.get_id())
    new_step.set_type(db.ElementaryStepType.BARRIERLESS)
    # Update compound linkage in flask
    flask.set_compounds(reactants)
    flask.add_reaction(new_reaction.id())
    for c_id in reactants:
        compound = db.Compound(c_id, compounds)
        compound.add_reaction(new_reaction.id())


def _create_compound(
    max_structures: int,
    properties: db.Collection,
    compounds: db.Collection,
    structures: db.Collection,
    user_input: bool = False,
):
    c = db.Compound(db.ID())
    c.link(compounds)
    c.create([])
    for _ in range(random.randint(1, max_structures)):
        c.add_structure(_fake_structure(c, structures, properties, user_input))
    c.disable_exploration()

    return c.get_id()


def _fake_structure(
    compound: db.Compound,
    structures: db.Collection,
    properties: db.Collection,
    user_input: bool,
):
    # Add structure data
    structure = db.Structure()
    structure.link(structures)
    structure.create(os.path.join(resources_root_path(), "water.xyz"), 0, 1)
    if user_input:
        structure.set_label(db.Label.USER_OPTIMIZED)
    else:
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
    structure.set_aggregate(compound.get_id())
    structure.set_model(db.Model("FAKE", "FAKE", "F-AKE"))
    add_random_energy(structure, (-20.0, -10.0), properties)

    return structure.get_id()


def _create_reaction(
    n_compounds: int,
    max_s_per_c: int,
    max_r_per_c: int,
    max_n_educts_per_r: int,
    max_n_products_per_r: int,
    max_steps_per_r: int,
    barrier_limits: Tuple[float, float],
    reactions: db.Collection,
    compounds: db.Collection,
    properties: db.Collection,
    structures: db.Collection,
    elementary_steps: db.Collection,
):
    success = False
    while not success:
        n_educts = random.randint(1, max_n_educts_per_r)
        verified_educts: List[db.ID] = []
        while not verified_educts:
            educt_objects = compounds.random_select_compounds(n_educts)
            for e in educt_objects:
                e.link(compounds)
                if len(e.get_reactions()) < max_r_per_c:
                    verified_educts.append(e.get_id())
        educts = verified_educts

        n_products = random.randint(1, max_n_products_per_r)
        current_n_compounds = compounds.count("{}")
        # pick existing compounds in database based on probability that existing would be picked from all planned
        # compounds
        n_existing_pick = int(round(n_products * current_n_compounds / n_compounds))
        verified_products: List[db.ID] = []
        while n_existing_pick > 0 and not verified_products:
            product_objects = compounds.random_select_compounds(n_existing_pick)
            for p in product_objects:
                p.link(compounds)
                if len(p.get_reactions()) < max_r_per_c and not any(educt == p.get_id() for educt in educts):
                    verified_products.append(p.get_id())
        # create new compounds to fill up reaction with new ones
        for _ in range(n_products - n_existing_pick):
            verified_products.append(_create_compound(max_s_per_c, properties, compounds, structures))
        products = verified_products

        # check if identical reaction already exists
        educt_types = [db.CompoundOrFlask.COMPOUND for i in educts]
        product_types = [db.CompoundOrFlask.COMPOUND for i in products]
        success = bool(identical_reaction(educts, products, educt_types, product_types, reactions) is None)

    reaction = db.Reaction()
    reaction.link(reactions)
    reaction.create(educts, products)
    for compound_id in educts + products:
        compound = db.Compound(compound_id, compounds)
        compound.add_reaction(reaction.get_id())

    steps = [
        _add_step(reaction, barrier_limits, compounds, structures, elementary_steps, properties)
        for _ in range(random.randint(1, max_steps_per_r))
    ]
    reaction.set_elementary_steps(steps)


def _redo_elementary_steps(
    reaction: db.Reaction,
    max_steps_per_r: int,
    barrier_limits: Tuple[float, float],
    compounds: db.Collection,
    structures: db.Collection,
    elementary_steps: db.Collection,
    properties: db.Collection,
):
    # delete old steps
    step_ids = reaction.get_elementary_steps()
    for sid in step_ids:
        step = db.ElementaryStep(sid)
        step.link(elementary_steps)
        # delete TS
        ts = db.Structure(step.get_transition_state())
        ts.link(structures)
        ts.wipe()
        step.wipe()
    # make new steps
    steps = [
        _add_step(reaction, barrier_limits, compounds, structures, elementary_steps, properties)
        for _ in range(random.randint(1, max_steps_per_r))
    ]
    reaction.set_elementary_steps(steps)


def _add_step(
    reaction: db.Reaction,
    barrier_limits: Tuple[float, float],
    compounds: db.Collection,
    structures: db.Collection,
    elementary_steps: db.Collection,
    properties: db.Collection,
) -> db.ID:
    step = db.ElementaryStep()
    step.link(elementary_steps)
    model = db.Model("FAKE", "FAKE", "F-AKE")
    compound_sides = reaction.get_reactants(db.Side.BOTH)
    # pick random structure for each compound
    step_structures = []
    for side in compound_sides:
        side_structures = []
        for c_id in side:
            c = db.Compound(c_id)
            c.link(compounds)
            side_structures.append(random.choice(c.get_structures()))
        step_structures.append(side_structures)
    step.create(*step_structures)
    reactant_energies = sum(
        [
            _get_electronic_energy(db.Structure(reactant), structures, properties, model)
            for reactant in step_structures[0]
        ]
    )
    product_energies = sum(
        [
            _get_electronic_energy(db.Structure(reactant), structures, properties, model)
            for reactant in step_structures[1]
        ]
    )
    shift = max(reactant_energies, product_energies)
    shifted_energy_limits = tuple(barrier + shift for barrier in barrier_limits)
    assert len(shifted_energy_limits) == 2
    # create TS
    ts = db.Structure()
    ts.link(structures)
    ts.create(os.path.join(resources_root_path(), "water.xyz"), 0, 1)
    ts.set_label(db.Label.TS_OPTIMIZED)
    ts.set_model(model)
    step.set_transition_state(ts.get_id())
    step.set_reaction(reaction.get_id())
    add_random_energy(ts, shifted_energy_limits, properties)  # type: ignore

    return step.get_id()


def add_random_energy(
    structure: db.Structure,
    energy_limits: Tuple[float, float],
    properties: db.Collection,
):
    """
    Adds a random electronic energy property to the given structure within the given limits.

    Parameters
    ----------
    structure :: db.Structure
        The Structure to add the energy to
    energy_limits :: Tuple[float, float]
        The lowest and highest possible energy in kJ/mol
    properties :: db.Collection
        The properties collection of the database
    """
    prop = db.NumberProperty()
    prop.link(properties)
    prop.create(db.Model("FAKE", "FAKE", "F-AKE"), "electronic_energy",
                random.uniform(*energy_limits) * utils.HARTREE_PER_KJPERMOL)
    structure.add_property(prop.get_property_name(), prop.get_id())

    return prop.get_id()


def _get_electronic_energy(
    structure: db.Structure,
    structures: db.Collection,
    properties_coll: db.Collection,
    model: db.Model = db.Model("FAKE", "FAKE", "F-AKE"),
) -> float:
    structure.link(structures)
    properties = structure.query_properties("electronic_energy", model, properties_coll)
    if not properties:
        raise RuntimeError("Missing requested electronic energy!")
    # pick last property if multiple
    prop = db.NumberProperty(properties[-1])
    prop.link(properties_coll)
    return prop.get_data() * utils.KJPERMOL_PER_HARTREE


def insert_single_empty_structure_compound(manager: db.Manager, label: db.Label) -> Tuple[db.ID, db.ID]:
    """
    Adds a structure and corresponding compound to the database, only use for testing!

    Parameters
    ----------
    manager :: db.Manager
        The database manager
    label :: db.Label
        The label for the structure of the compound
    -------
    compound, structure :: Tuple[db.ID, db.ID]
        The IDs of the inserted compound and structure
    """
    structures = manager.get_collection("structures")
    compounds = manager.get_collection("compounds")
    structure = db.Structure()
    structure.link(structures)
    rr = resources_root_path()
    structure.create(os.path.join(rr, "water.xyz"), 0, 1)
    compound = db.Compound()
    compound.link(compounds)
    compound.create([structure.get_id()])
    compound.disable_exploration()
    structure.set_aggregate(compound.get_id())
    structure.set_label(label)

    return compound.get_id(), structure.get_id()


def test_random_db():
    n_compounds = 9
    n_reactions = 6
    n_flasks = 1
    max_r_per_c = 10
    max_n_products_per_r = 2
    max_n_educts_per_r = 2
    max_s_per_c = 2
    max_steps_per_r = 1
    barrier_limits = (0.1, 100.0)
    n_inserts = 3
    manager = get_random_db(
        n_compounds,
        n_flasks,
        n_reactions,
        max_r_per_c,
        "chemoton_test_random_db",
        max_n_products_per_r,
        max_n_educts_per_r,
        max_s_per_c,
        max_steps_per_r,
        barrier_limits,
        n_inserts,
    )

    compounds = manager.get_collection("compounds")
    reactions = manager.get_collection("reactions")
    structures = manager.get_collection("structures")
    elementary_steps = manager.get_collection("elementary_steps")

    assert compounds.count(dumps({})) == n_compounds
    assert reactions.count(dumps({})) == n_reactions
    n_steps = elementary_steps.count(dumps({}))
    assert n_steps <= max_steps_per_r * n_reactions
    assert structures.count(dumps({})) <= n_compounds * max_s_per_c + n_steps  # each step creates a TS structure
    for structure in structures.iterate_all_structures():
        structure.link(structures)
        assert structure.has_property("electronic_energy")

    import pytest
    with pytest.raises(AssertionError):
        _ = get_random_db(5, 1, 10, 100, max_n_products_per_r=10)
    with pytest.raises(AssertionError):
        _ = get_random_db(50, 50, 1, 100)
