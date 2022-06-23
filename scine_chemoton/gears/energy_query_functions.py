#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import Union
# Third party imports
import scine_database as db
import scine_utilities as utils


def get_single_barrier_for_elementary_step(step: db.ElementaryStep, model: db.Model, structures: db.Collection,
                                           properties: db.Collection) -> Union[float, None]:
    """
    Gives the barrier height of a single elementary step (left to right) in kJ/mol. If available, the gibbs free energy
    ('gibbs_free_energy') barrier is returned. Otherwise the electronic energy ('electronic_energy') barrier is
    returned. Returns None if both energies are unavailable.

    Parameters
    ----------
    step : scine_database.ElementaryStep (Scine::Database::ElementaryStep)
        The elementary step we want the barrier height from
    model : scine_database.Model
        The model used to calculate the energies.
    structures : scine_database.Collection
        The structure collection.
    properties : scine_database.Collection
        The property collection.

    Returns
    -------
    Union[float, None]
        barrier in kJ/mol
    """
    gibbs = get_single_barrier_for_elementary_step_by_type(step, "gibbs_free_energy", model, structures, properties)
    if gibbs is not None:
        return gibbs
    else:
        return get_single_barrier_for_elementary_step_by_type(step, "electronic_energy", model, structures, properties)


def get_single_barrier_for_elementary_step_by_type(step: db.ElementaryStep, energy_type: str, model: db.Model,
                                                   structures: db.Collection, properties: db.Collection) \
        -> Union[float, None]:
    """
    Gives the barrier height of a single elementary step (left to right) in kJ/mol for the specified energy type.
    Returns None if the energy type is not available.

    Parameters
    ----------
    step : scine_database.ElementaryStep (Scine::Database::ElementaryStep)
        The elementary step we want the barrier height from
    energy_type : str
        The name of the energy property such as 'electronic_energy' or 'gibbs_free_energy'
    model : scine_database.Model
        The model used to calculate the energies.
    structures : scine_database.Collection
        The structure collection.
    properties : scine_database.Collection
        The property collection.

    Returns
    -------
    Union[float, None]
        barrier in kJ/mol
    """
    ts = db.Structure(step.get_transition_state())
    ts_energy = get_energy_for_structure(ts, energy_type, model, structures, properties)
    reactant_energies = [
        get_energy_for_structure(db.Structure(reactant), energy_type, model, structures, properties)
        for reactant in step.get_reactants(db.Side.LHS)[0]
    ]
    reactant_energy = None if None in reactant_energies else sum(reactant_energies)
    return (
        None if None in [ts_energy, reactant_energy] else (ts_energy - reactant_energy) * utils.KJPERMOL_PER_HARTREE
    )


def get_energy_for_structure(structure: db.Structure, prop_name: str, model: db.Model, structures: db.Collection,
                             properties: db.Collection) -> Union[float, None]:
    """
    Gives energy value depending on demanded property. If the property does not exit, None is returned.

    Parameters
    ----------
    structure : scine_database.Structure (Scine::Database::Structure)
        The structure we want the energy from
    prop_name : str
        The name of the energy property such as 'electronic_energy' or 'gibbs_free_energy'
    model : scine_database.Model
        The model used to calculate the energies.
    structures : scine_database.Collection
        The structure collection.
    properties : scine_database.Collection
        The property collection.

    Returns
    -------
    Union[float, None]
        energy value in Hartree
    """
    structure.link(structures)
    structure_properties = structure.query_properties(prop_name, model, properties)
    if not structure_properties:
        return None
    # pick last property if multiple
    prop = db.NumberProperty(structure_properties[-1])
    prop.link(properties)
    return prop.get_data()
