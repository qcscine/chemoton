#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Union, List, Tuple, Optional
import warnings

import scine_database as db
from scine_database.insert_concentration import insert_concentration_for_structure
import scine_utilities as utils
from pymatgen.io.cif import CifParser

from .surfaces.pymatgen_interface import PmgInterface


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
        Label of the inserted structure, by default db.Label.USER_GUESS.
    job : db.Job, optional
        Job to be performed on the initial structure, by default db.Job('scine_geometry_optimization').
    settings : utils.ValueCollection, optional
        Job settings, by default none.
    start_concentration : float
        The start concentration of the compound that will be generated from this structure.

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


def insert_surface_from_materials_project(database: db.Manager, materials_project_query: str,
                                          miller: Union[str, int, List[int]],
                                          charge: int, multiplicity: int, model: db.Model,
                                          label: db.Label = db.Label.USER_GUESS,
                                          job: db.Job = db.Job('scine_geometry_optimization'),
                                          settings: Optional[dict] = None, slab_settings: Optional[dict] = None,
                                          extension: int = 1, conventional_cell: bool = True,
                                          query_type: str = 'chemical_formula') \
        -> List[Tuple[db.Structure, db.Calculation]]:

    if query_type == 'chemical_formula':
        crystal = PmgInterface.get_crystal_structure_by_formula(materials_project_query, conventional_cell)
    elif query_type == 'material_id':
        crystal = PmgInterface.get_crystal_structure_by_id(materials_project_query, conventional_cell)
    else:
        raise NotImplementedError("Only support query by 'chemical_formula' and 'material_id'")

    if slab_settings is None:
        slab_settings = {}
    slabs = PmgInterface.get_slabs(crystal, miller, **slab_settings)
    return PmgInterface.insert_slabs(slabs, charge, multiplicity, model, label, job, settings, extension, database)


def insert_surface_from_cif_file(database: db.Manager, crystal_structure_filename: str,
                                 miller: Union[str, int, List[int]],
                                 charge: int, multiplicity: int, model: db.Model,
                                 label: db.Label = db.Label.USER_GUESS,
                                 job: db.Job = db.Job('scine_geometry_optimization'),
                                 settings: Optional[dict] = None, slab_settings: Optional[dict] = None,
                                 extension: int = 1) \
        -> List[Tuple[db.Structure, db.Calculation]]:
    parser = CifParser(crystal_structure_filename)
    if parser.has_errors:
        for warning in parser.warnings:
            print(warning)
    pymatgen_structure = parser.get_structures()[0]

    if slab_settings is None:
        slab_settings = {}
    slabs = PmgInterface.get_slabs(pymatgen_structure, miller, **slab_settings)
    return PmgInterface.insert_slabs(slabs, charge, multiplicity, model, label, job, settings, extension, database)
