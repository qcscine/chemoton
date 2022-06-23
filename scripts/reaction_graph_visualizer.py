#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualizes all reactions in the database with molassembler graphs.
   Runs at least with the native IPython support of vscode"""

__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""
#%%
import scine_database as db
import scine_molassembler as masm

from IPython import display

manager = db.Manager()
db_name = "testdb"
db_port = 27017
credentials = db.Credentials("127.0.0.1", db_port, db_name)
manager.set_credentials(credentials)
manager.connect()
structures = manager.get_collection("structures")
calculations = manager.get_collection("calculations")
properties = manager.get_collection("properties")
compounds = manager.get_collection("compounds")
reactions = manager.get_collection("reactions")
elementary_steps = manager.get_collection("elementary_steps")


def cbor_to_mol(cbor_str: str) -> masm.Molecule:
    """Reconvert a cbor serialization to a molassembler molecule

    Parameters
    ----------
    cbor_graph : str
        [description]

    Returns
    -------
    masm.Molecule
        [description]
    """
    binary = masm.JsonSerialization.base_64_decode(cbor_str)
    serializer = masm.JsonSerialization(binary, masm.JsonSerialization.BinaryFormat.CBOR)
    return serializer.to_molecule()


#%% Illustrate all reactions
# Loops over all reactions and visualizes the involved molecules as graphs
# Press enter to proceed to next reaction or 'q' to leave.
for reaction in reactions.iterate_all_reactions():
    print("*** Visualising reaction {}".format(reaction.get_id().string()))
    reaction.link(reactions)
    side_count = 0
    counter = 0
    for side_compounds in reaction.get_reactants(db.Side.BOTH):
        if side_count:
            print("-->")
        side_count += 1
        counter += 1
        side_images = []
        side_list = []

        for reactant_compound_id in side_compounds:
            reactant_compound = db.Compound(reactant_compound_id)
            reactant_compound.link(compounds)
            reactant_id = reactant_compound.get_centroid()
            reactant = db.Structure(reactant_id)
            reactant.link(structures)
            cbor_graph = reactant.get_graph("masm_cbor_graph")
            mol = cbor_to_mol(cbor_graph)
            canonicalization = mol.canonicalize()
            display.display_svg(mol)

    stop = input("Press enter to proceed to next reaction or 'q' to leave.")
    if stop == "q":
        break
    display.clear_output()

# %%
