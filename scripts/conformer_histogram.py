#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import scine_utilities as utils
import scine_database as db
import scine_molassembler as masm

import sys

import matplotlib.pyplot as plt

from json import dumps


def get_energies(structures, properties):
    # Setup query for optimized structures without compound
    selection = {
        "$and": [
            {"label": {"$eq": "minimum_optimized"}},
            {"compound": {"$oid": compound_id}},
            {"properties.electronic_energy": {"$exists": True}},
            {"properties.gibbs_energy": {"$exists": True}},
        ]
    }
    # Loop over all results
    electronic_energies = []
    gibbs_energies = []
    for structure in structures.iterate_structures(dumps(selection)):
        structure.link(structures)
        electronic_energy = db.NumberProperty(structure.get_properties("electronic_energy")[0])
        electronic_energy.link(properties)
        electronic_energies.append(electronic_energy.get_data() * 2625.5)
        gibbs_energy = db.NumberProperty(structure.get_properties("gibbs_free_energy")[0])
        gibbs_energy.link(properties)
        gibbs_energies.append(gibbs_energy.get_data() * 2625.5)

    min_el = min(electronic_energies)
    min_gibbs = min(gibbs_energies)

    return [e - min_el for e in electronic_energies], [e - min_gibbs for e in gibbs_energies]


if __name__ == "__main__":

    # Prepare clean database
    manager = db.Manager()
    credentials = db.Credentials("127.0.0.1", 27017, "puffin_tests")
    manager.set_credentials(credentials)
    manager.connect()

    model = db.Model("pm6", "", "")
    # model = db.Model('dft','pbe-d3bjabc','def2-svp')
    opt_settings = {"convergence_max_iterations": "500"}

    if len(sys.argv) < 2:
        print("Compound ID required")
        exit()

    if "ObjectId" in sys.argv[1]:
        compound_id = sys.argv[1].split('"')[1]
    else:
        compound_id = sys.argv[1]

    structures = manager.get_collection("structures")
    properties = manager.get_collection("properties")

    electronic_energies, gibbs_energies = get_energies(structures, properties)

    plt.subplot(2, 1, 1)
    plt.hist(electronic_energies, bins="auto", color="lightblue")
    plt.title("Electronic Energy Distribution")
    plt.xlabel("Relative Electronic Energy in kJ/mol")
    plt.ylabel("Count")

    plt.subplot(2, 1, 2)
    plt.hist(gibbs_energies, bins="auto", color="lightblue")
    plt.title("Gibbs Energy Distribution")
    plt.xlabel("Relative Gibbs Energy in kJ/mol")
    plt.ylabel("Count")

    plt.subplots_adjust(hspace=0.5)
    plt.savefig("histogram.pdf")

    while "loop" in sys.argv:
        electronic_energies, gibbs_energies = get_energies(structures, properties)
        plt.subplot(2, 1, 1)
        plt.hist(electronic_energies, bins="auto", color="lightblue")
        plt.subplot(2, 1, 2)
        plt.hist(gibbs_energies, bins="auto", color="lightblue")
        plt.pause(1.0)
