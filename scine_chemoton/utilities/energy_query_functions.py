#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import Union, Tuple
# Third party imports
import scine_database as db
import scine_utilities as utils


def get_energy_sum_of_elementary_step_side(step: db.ElementaryStep, side: db.Side, energy_type: str, model: db.Model,
                                           structures: db.Collection, properties: db.Collection) -> Union[float, None]:
    """
    Gives the total energy in atomic units of the given side of the step. Returns None if the energy type is
    not available.

    Parameters
    ----------
    step : scine_database.ElementaryStep (Scine::Database::ElementaryStep)
        The elementary step we want the energy from
    side : scine_database.Side (Scine::Database::Side)
        The side we want the side from
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
        Energy in hartree
    """
    if side == db.Side.BOTH:
        raise RuntimeError("The energy sum of both sides of a step is not supported.")
    index = 0 if side == db.Side.LHS else 1
    energies = [
        get_energy_for_structure(db.Structure(reactant), energy_type, model, structures, properties)
        for reactant in step.get_reactants(side)[index]
    ]
    return None if None in energies else sum(energies)  # type: ignore


def get_barriers_for_elementary_step_by_type(step: db.ElementaryStep, energy_type: str, model: db.Model,
                                             structures: db.Collection, properties: db.Collection)\
        -> Union[Tuple[float, float], Tuple[None, None]]:
    """
    Gives the forward and backward barrier height of a single elementary step (left to right) in kJ/mol for the
    specified energy type. Returns None if the energy type is not available.

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
    reactant_energy = get_energy_sum_of_elementary_step_side(step, db.Side.LHS, energy_type, model,
                                                             structures, properties)
    if reactant_energy is None:
        return None, None
    product_energy = get_energy_sum_of_elementary_step_side(step, db.Side.RHS, energy_type, model,
                                                            structures, properties)
    if product_energy is None:
        return None, None
    if step.get_type() == db.ElementaryStepType.BARRIERLESS:
        if product_energy > reactant_energy:
            return (product_energy - reactant_energy) * utils.KJPERMOL_PER_HARTREE, 0.0
        else:
            return 0.0, (reactant_energy - product_energy) * utils.KJPERMOL_PER_HARTREE
    ts = db.Structure(step.get_transition_state())
    ts_energy = get_energy_for_structure(ts, energy_type, model, structures, properties)
    if ts_energy is None:
        return None, None
    lhs_barrier = (ts_energy - reactant_energy) * utils.KJPERMOL_PER_HARTREE
    rhs_barrier = (ts_energy - product_energy) * utils.KJPERMOL_PER_HARTREE
    return lhs_barrier, rhs_barrier


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
    if len(structure_properties) < 1:
        return None
    # pick last property if multiple
    prop = db.NumberProperty(structure_properties[-1])
    prop.link(properties)
    return prop.get_data()


def rate_constant_from_barrier(barrier: float, temperature: float) -> float:
    """
    Calculate a rate constant from its energy and temperature according to transition state theory:
    rate-constant = k_B T / h exp[-barrier/(R T)]

    Parameters
    ----------
    barrier :: float
        The reaction barrier in kJ/mol.
    temperature :: flaot
        The temperature in K.

    Returns
    -------
    The rate constant.
    """
    from math import exp
    barrier_j_per_mol = 1e+3 * barrier
    kbt_in_j = utils.BOLTZMANN_CONSTANT * temperature  # k_B T
    factor = kbt_in_j / utils.PLANCK_CONSTANT  # k_B T / h
    rt_in_j_per_mol = utils.MOLAR_GAS_CONSTANT * temperature  # R T
    beta_in_mol_per_j = 1.0 / rt_in_j_per_mol  # 1 / (R T)
    return factor * exp(- beta_in_mol_per_j * barrier_j_per_mol)
