#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import scine_utilities as utils
import scine_database as db
import scine_molassembler as masm

import numpy as np
import sys

import matplotlib.pyplot as plt

from json import dumps


def get_runtimes(calculations):
    # Setup query for finished calculations
    selection = {
        "$and": [
            {"status": {"$eq": "complete"}},
        ]
    }
    # Loop over all results
    walltime = []
    cpuh = []
    for calculation in calculations.iterate_calculations(dumps(selection)):
        calculation.link(calculations)
        cores = calculation.get_job().cores
        runtime = calculation.get_runtime()
        walltime.append(runtime)
        cpuh.append(cores * runtime)

    return walltime, cpuh


if __name__ == "__main__":

    # Prepare clean database
    manager = db.Manager()
    credentials = db.Credentials("127.0.0.1", 27017, "puffin_tests")
    manager.set_credentials(credentials)
    manager.connect()

    model = db.Model("pm6", "", "")
    # model = db.Model('dft','pbe-d3bjabc','def2-svp')
    opt_settings = {"convergence_max_iterations": "500"}

    calculations = manager.get_collection("calculations")

    walltime, cpuh = get_runtimes(calculations)

    plt.subplot(2, 1, 1)
    MIN = np.floor(np.log10(min(walltime)))
    MAX = np.ceil(np.log10(max(walltime)))
    plt.hist(walltime, bins=10 ** np.linspace(MIN, MAX, 50), log=False, color="lightblue")
    plt.title("Walltime")
    plt.xlabel("Walltime in s")
    plt.ylabel("Count")
    plt.gca().set_xscale("log")

    plt.subplot(2, 1, 2)
    MIN = np.floor(np.log10(min(cpuh)))
    MAX = np.ceil(np.log10(max(cpuh)))
    plt.hist(cpuh, bins=10 ** np.linspace(MIN, MAX, 50), log=False, color="lightblue")
    plt.title("CPU Time")
    plt.xlabel("CPU Time in s")
    plt.ylabel("Count")
    plt.gca().set_xscale("log")

    plt.subplots_adjust(hspace=0.5)
    plt.savefig("runtime.pdf")

    while "loop" in sys.argv:
        walltime, cpuh = get_runtimes(calculations)
        plt.subplot(2, 1, 1)
        MIN = np.floor(np.log10(min(walltime)))
        MAX = np.ceil(np.log10(max(walltime)))
        plt.hist(walltime, bins=10 ** np.linspace(MIN, MAX, 50), log=False, color="lightblue")
        plt.subplot(2, 1, 2)
        MIN = np.floor(np.log10(min(cpuh)))
        MAX = np.ceil(np.log10(max(cpuh)))
        plt.hist(cpuh, bins=10 ** np.linspace(MIN, MAX, 50), log=False, color="lightblue")
        plt.pause(1.0)
