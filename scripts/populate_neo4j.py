#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# The graph DB "Neo4j can be downloaded from https://neo4j.com/. Note
# that there is a free community edition. Once downloaded, simply
# extract it and run "./bin/neo4j start" to start the DB.
#
# The company offers also a GUI program called "Neo4j Desktop" which
# is essentially a bundle of different GUIs and the Neo4j DB. Since it
# is distributed as an app image, you essentially just need to download
# a file and execute it; there is no complicated installation procedure
# and you don't need root rights. Unfortunately, it doesn't run on
# CentOS 7.
#
# However, we can make use of the web interface offered by the Neo4j
# DB in order to get a GUI. Upon first tests, it appears that this
# GUI might be powerful enough for our purposes; further tests will
# show whether this is really true.

# Note: This module works with Neo4j DB 3.5.23 but not with 4.0.8 or 4.1.3
from neo4jrestclient.client import GraphDatabase
from json import dumps
import scine_database as mongo_db
import scine_utilities as utils


def barrier(reaction, steps, structures, properties) -> float:
    barriers = []
    for step_id in reaction.get_elementary_steps():
        step = mongo_db.ElementaryStep(step_id)
        step.link(steps)
        ts = mongo_db.Structure(step.get_transition_state())
        ts_energy = energy(ts, structures, properties)
        reactant_energy = sum(
            energy(mongo_db.Structure(educt), structures, properties)
            for educt in step.get_reactants(mongo_db.Side.LHS)[0]
        )
        barriers.append(ts_energy - reactant_energy)
    return min(barriers) * utils.KJPERMOL_PER_HARTREE


def energy(structure, structures, properties) -> float:
    structure.link(structures)
    if not structure.has_property("electronic_energy"):
        raise RuntimeError("Missing energy data for barrier height. Disable option to generate network without it.")
    prop = mongo_db.NumberProperty(structure.get_property("electronic_energy"))
    prop.link(properties)
    return prop.get_data()


if __name__ == "__main__":
    calculate_barriers = True
    barrier_limit = 1000.0
    # Connect to MongoDB
    manager = mongo_db.Manager()
    credentials = mongo_db.Credentials("127.0.0.1", 27017, DB_NAME)
    manager.set_credentials(credentials)
    manager.connect()

    # Connect to Neo4j DB
    uri = "http://localhost:7474"  # This is the default connection (this is also the address for the web interface)
    username = "neo4j"  # This is the default username
    password = "password"  # This has to be set when first connecting via the web interface
    neo4j_db = GraphDatabase(uri, username=username, password=password)

    # Read out reactions from MongoDB and populate Neo4j DB accordingly
    mongodb_reactions = manager.get_collection("reactions")
    mongodb_compounds = manager.get_collection("compounds")
    mongodb_structures = manager.get_collection("structures")
    mongodb_properties = manager.get_collection("properties")
    mongodb_elementary_steps = manager.get_collection("elementary_steps")

    neo4j_compounds = neo4j_db.labels.create("Compound")
    if calculate_barriers:
        neo4j_high_reactions = neo4j_db.labels.create("High barrier Reaction")
        neo4j_reactions = neo4j_db.labels.create("Reaction")
    else:
        neo4j_reactions = neo4j_db.labels.create("Reaction")

    for count, mongodb_reaction in enumerate(mongodb_reactions.iterate_all_reactions(), 1):
        mongodb_reaction.link(mongodb_reactions)
        reactants, products = mongodb_reaction.get_reactants(mongo_db.Side.BOTH)
        # add reaction
        if calculate_barriers:
            barrier_height = barrier(
                mongodb_reaction,
                mongodb_elementary_steps,
                mongodb_structures,
                mongodb_properties,
            )
            color = "green" if barrier_height < barrier_limit else "red"
            neo4j_reaction = neo4j_db.nodes.create(name="reaction_" + str(count), barrier=barrier_height, color=color)
            if barrier_height < barrier_limit:
                neo4j_reactions.add(neo4j_reaction)
            else:
                neo4j_high_reactions.add(neo4j_reaction)
        else:
            neo4j_reaction = neo4j_db.nodes.create(name="reaction_" + str(count))
            neo4j_reactions.add(neo4j_reaction)
        # add reactants
        for reactant in reactants:
            hits = neo4j_compounds.get(name=reactant.string())
            if len(hits) == 0:
                mongodb_compound = mongo_db.Compound(reactant)
                mongodb_compound.link(mongodb_compounds)
                lhs = neo4j_db.nodes.create(name=reactant.string(), explore=str(mongodb_compound.explore()))
                neo4j_compounds.add(lhs)
            elif len(hits) == 1:
                lhs = hits[0]
            else:
                raise RuntimeError("Something went wrong, identical IDs for different objects.")
            lhs.relationships.create("reacts", neo4j_reaction)
        # add products
        for product in products:
            hits = neo4j_compounds.get(name=product.string())
            if len(hits) == 0:
                mongodb_compound = mongo_db.Compound(product)
                mongodb_compound.link(mongodb_compounds)
                rhs = neo4j_db.nodes.create(name=product.string(), explore=str(mongodb_compound.explore()))
                neo4j_compounds.add(rhs)
            elif len(hits) == 1:
                rhs = hits[0]
            else:
                raise RuntimeError("Something went wrong, identical IDs for different objects.")
            neo4j_reaction.relationships.create("produces", rhs)
    # add compounds without reaction (make sure that compounds have reactions linked (bug in versions pre Jan 2021))
    selection = {"reactions": {"$exists": True, "$eq": []}}
    for mongodb_compound in mongodb_compounds.query_compounds(dumps(selection)):
        mongodb_compound.link(mongodb_compounds)
        compound = neo4j_db.nodes.create(
            name=mongodb_compound.get_id().string(),
            explore=str(mongodb_compound.explore()),
        )
        neo4j_compounds.add(compound)

# Alternatively, the Python module "neo4j" works with Neo4j 4.1.3.
# It's usage is easy, but more involved, mostly because queries
# have to be written manually. For example, an example reaction
# A + B -> C + D would be realized with the following CQL query:
cql_create = """CREATE (rhs1:compound { name: "23562e"}),
(rhs2:compound { name: "5684h6"}),
(lhs1:compound { name: "h78kl0"}),
(lhs2:compound { name: "659h90"}),
(reaction:reaction {name: "reaction1"}),(lhs1)-[:reacts]->(reaction),
(lhs2)-[:reacts]->(reaction),
(reaction)-[:reacts]->(rhs1),
(reaction)-[:reacts]->(rhs2)"""
