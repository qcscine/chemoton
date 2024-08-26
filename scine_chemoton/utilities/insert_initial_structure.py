#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Union, Optional
import warnings

import scine_database as db
from scine_database.insert_concentration import insert_concentration_for_structure
import scine_utilities as utils


def insert_initial_structure(
    database: db.Manager,
    molecule_path: Union[str, utils.AtomCollection],
    charge: int,
    multiplicity: int,
    model: db.Model,
    label: db.Label = db.Label.USER_GUESS,
    job: db.Job = db.Job("scine_geometry_optimization"),
    settings: utils.ValueCollection = utils.ValueCollection({}),
    start_concentration: Optional[float] = None
):
    """
    Insert a structure to the database and set up a calculation working on it.

    Parameters
    ----------
    database : db.Manager
        Database to use.
    molecule_path : Union[str, utils.AtomCollection]
        Atom collection or path to the xyz file with the structure to be inserted.
    charge : int
        Charge of the structure.
    multiplicity : int
        Multiplicity of the structure.
    model : db.Model
        Model to be used for the calculation.
    label : db.Label, optional
        Label of the inserted structure, by default db.Label.MINIMUM_GUESS.
    job : db.Job, optional
        Job to be performed on the initial structure, by default db.Job('scine_geometry_optimization').
    settings : utils.ValueCollection, optional
        Job settings, by default none.
    start_concentration : float
        The start concentratoin of the compound that will be generated from this structure.

    Returns
    -------
    db.Structure, db.Calculation
        The inserted structure and the calculation generated for it
    """
    structures = database.get_collection("structures")
    calculations = database.get_collection("calculations")

    structure = db.Structure()
    structure.link(structures)
    structure.create(molecule_path, charge, multiplicity)
    if label != db.Label.USER_GUESS:
        warnings.warn(
            "WARNING: You specified a label for your structure input that is not 'user_guess'. This may "
            "hinder the exploration of this structure."
        )
    structure.set_label(label)

    if start_concentration is not None:
        insert_concentration_for_structure(database, start_concentration, model, structure.id())

    if label == db.Label.USER_OPTIMIZED:
        structure.set_model(model)
        return structure, None

    calculation = db.Calculation()
    calculation.link(calculations)
    calculation.create(model, job, [structure.id()])
    calculation.set_priority(1)

    if settings:
        calculation.set_settings(settings)

    calculation.set_status(db.Status.NEW)
    return structure, calculation
