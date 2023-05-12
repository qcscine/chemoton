#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import Union, Tuple, List, Optional
import numpy as np
# Third party imports
import scine_database as db
import scine_utilities as utils


def get_elementary_step_with_min_ts_energy(reaction: db.Reaction,
                                           energy_type: str,
                                           model: db.Model,
                                           elementary_steps: db.Collection,
                                           structures: db.Collection,
                                           properties: db.Collection,
                                           structure_model: Optional[db.Model] = None) -> Optional[db.ID]:
    """
    Gets the elementary step ID with the lowest energy of the corresponding transition state of a reaction.

    Parameters
    ----------
    reaction : db.Reaction
        The reaction for which the elementary steps shall be analyzed.
    energy_type : str
        The name of the energy property such as 'electronic_energy' or 'gibbs_free_energy'
    model : scine_database.Model
        The model used to calculate the energies.
    elementary_steps : db.Collection
        The elementary step collection.
    structures : scine_database.Collection
        The structure collection.
    properties : scine_database.Collection
        The property collection.
    structure_model : Optional[db.Model]
        The model of the transition state. If None, the model of the transition state is not checked.

    Returns
    -------
    Optional[db.ID]
        The ID of the elementary step with the lowest TS energy.
    """
    lowest_ts_energy = np.inf
    es_id_with_lowest_ts = None
    # # # Loop over elementary steps
    for es_id in reaction.get_elementary_steps():
        es = db.ElementaryStep(es_id, elementary_steps)
        # # # Type check elementary step and break if barrierless
        if es.get_type() == db.ElementaryStepType.BARRIERLESS:
            first_structure_lhs = db.Structure(es.get_reactants()[0][0], structures)
            if structure_model is not None and first_structure_lhs.get_model() != structure_model:
                continue
            # # # Energy Check for minima
            first_structure_lhs_energy = get_energy_for_structure(first_structure_lhs, energy_type, model,
                                                                  structures, properties)
            if first_structure_lhs_energy is None:
                continue
            es_id_with_lowest_ts = es_id
            break
        ts = db.Structure(es.get_transition_state(), structures)
        if structure_model is not None and ts.get_model() != structure_model:
            continue

        # # # Costly safety check that barrier as well as ts_energy exist for this model and energy type
        ts_energy = get_energy_for_structure(
            ts, energy_type, model, structures, properties)
        barriers = get_barriers_for_elementary_step_by_type(es, energy_type, model, structures, properties)
        if None in barriers or ts_energy is None:
            continue
        # # # Comparison with current lowest energy
        if ts_energy < lowest_ts_energy:
            es_id_with_lowest_ts = es_id
            lowest_ts_energy = ts_energy

    return es_id_with_lowest_ts


def get_energy_sum_of_elementary_step_side(step: db.ElementaryStep, side: db.Side, energy_type: str, model: db.Model,
                                           structures: db.Collection, properties: db.Collection) -> Optional[float]:
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
    Optional[float]
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
    Calculate a rate constant from its energy [kJ / mol] and temperature according to transition state theory:
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
    barrier_j_per_mol = 1e+3 * barrier
    kbt_in_j = utils.BOLTZMANN_CONSTANT * temperature  # k_B T
    factor = kbt_in_j / utils.PLANCK_CONSTANT  # k_B T / h
    rt_in_j_per_mol = utils.MOLAR_GAS_CONSTANT * temperature  # R T
    beta_in_mol_per_j = 1.0 / rt_in_j_per_mol  # 1 / (R T)
    return factor * np.exp(- beta_in_mol_per_j * barrier_j_per_mol)


def get_all_energies_for_aggregate(aggregate: Union[db.Compound, db.Flask], model: db.Model, energy_label: str,
                                   structures: db.Collection, properties: db.Collection) -> List[Union[float, None]]:
    all_energies = []
    for s_id in aggregate.get_structures():
        structure = db.Structure(s_id)
        all_energies.append(get_energy_for_structure(structure, energy_label, model, structures, properties))
    return all_energies


def get_min_energy_for_aggregate(aggregate: Union[db.Compound, db.Flask], model: db.Model, energy_label: str,
                                 structures: db.Collection, properties: db.Collection) -> Optional[float]:
    all_energies = get_all_energies_for_aggregate(aggregate, model, energy_label, structures, properties)
    if len(all_energies) < 1:
        return None
    non_none_energies = list()
    for e in all_energies:
        if e is not None:
            non_none_energies.append(e)
    if len(non_none_energies) < 1:
        return None
    return min(non_none_energies)


def get_min_free_energy_for_aggregate(aggregate: Union[db.Compound, db.Flask], electronic_model: db.Model,
                                      correction_model: db.Model, structures: db.Collection,
                                      properties: db.Collection) -> Optional[float]:
    equal_models = electronic_model == correction_model
    if equal_models:
        return get_min_energy_for_aggregate(aggregate, electronic_model, "gibbs_free_energy", structures, properties)
    all_energies: List[float] = []
    for s_id in aggregate.get_structures():
        structure = db.Structure(s_id)
        electronic_energy = get_energy_for_structure(structure, "electronic_energy", electronic_model, structures,
                                                     properties)
        free_energy_correction = get_energy_for_structure(structure, "gibbs_energy_correction", correction_model,
                                                          structures, properties)
        if electronic_energy is None or free_energy_correction is None:
            continue
        energy = electronic_energy + free_energy_correction
        all_energies.append(energy)
    if not all_energies:
        return None
    return min(all_energies)
