#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Union, List

# Third party imports
import scine_database as db

from ...utilities.compound_and_flask_creation import get_compound_or_flask


def query_concentration_with_model_id(label: str, compound_id: db.ID, compounds: db.Collection,
                                      properties: db.Collection, structures: db.Collection, model: db.Model) -> float:
    """
    Query a concentration property with a given label and electronic structure model for the compound id.

    Parameters
    ----------
    label : str
        The concentration property label.
    compound_id : db.ID
        The compound_id.
    compounds : db.Collection
        The compound collection.
    properties : db.Collection
        The property collection.
    structures : db.Collection
        The structure collection.
    model : db.Model
        The electronic structure model that characterizes the property.

    Returns
    -------
    float
        The concentration according to property label and electronic structure model. Return 0.0 if no property is
        present.
    """
    compound = db.Compound(compound_id, compounds)
    return query_concentration_with_model_object(label, compound, properties, structures, model)


def query_concentration_with_model_object(label: str, compound: Union[db.Compound, db.Flask],
                                          properties: db.Collection, structures: db.Collection, model: db.Model)\
        -> float:
    """
    Query a concentration property with a given label and electronic structure model for the compound.

    Parameters
    ----------
    label : str
        The concentration property label.
    compound : db.Compound
        The compound as a linked object.
    properties : db.Collection
        The property collection.
    structures : db.Collection
        The structure collection.
    model : db.Model
        The electronic structure model that characterizes the property.

    Returns
    -------
    float
        The concentration according to property label and electronic structure model. Return 0.0 if no property is
        present.
    """
    centroid = db.Structure(compound.get_centroid(), structures)
    property_list = centroid.query_properties(label, model, properties)
    if not property_list:
        return 0.0
    # pick last property if multiple
    prop = db.NumberProperty(property_list[-1], properties)
    return prop.get_data()


def query_concentration_with_object(label: str, compound: Union[db.Compound, db.Flask],
                                    properties: db.Collection, structures: db.Collection) -> float:
    """
    Query a concentration property with a given label. The last property with given label is used independent of
    the electronic structure model.

    Parameters
    ----------
    label : str
        The concentration property label.
    compound : db.Compound
        The compound as a linked object.
    properties : db.Collection
        The property collection.
    structures : db.Collection
        The structure collection.

    Returns
    -------
    float
        The concentration according to property label and electronic structure model. Return 0.0 if no property is
        present.
    """
    centroid = db.Structure(compound.get_centroid(), structures)

    property_list = centroid.get_properties(label)
    if not property_list:
        return 0.0
    # pick last property if multiple
    prop = db.NumberProperty(property_list[-1], properties)
    return prop.get_data()


def query_reaction_flux_with_model(label_post_fix: str, reaction: db.Reaction, compounds: db.Collection,
                                   flasks: db.Collection, structures: db.Collection, properties: db.Collection,
                                   model: db.Model) -> float:
    """
    Query the absolute flux along the reaction.
    Parameters
    ----------
    label_post_fix: str
        Post-fix for the concentration label. The property label will be given as reaction.id() + label_post_fix
    reaction: db.Reaction
        The reaction.
    compounds: db.Collection
        The compound collection.
    flasks: db.Collection
        The flask collection.
    structures: db.Collection
        The structure collection.
    properties: db.Collection
        The property collection.
    model: db.Model
        The electronic structure model which must be tagged on the concentration property.
    """
    label = reaction.id().string() + label_post_fix
    a_id = reaction.get_reactants(db.Side.LHS)[0][0]
    a_type = reaction.get_reactant_types(db.Side.LHS)[0][0]
    aggregate = get_compound_or_flask(a_id, a_type, compounds, flasks)
    return query_concentration_with_model_object(label, aggregate, properties, structures, model)


def query_concentrations_with_model(label: str, aggregate: Union[db.Compound, db.Flask], properties: db.Collection,
                                    structures: db.Collection, model: db.Model) -> List[float]:
    """
    Query all concentrations with a given label and electronic structure model.

    Parameters
    ----------
    label : str
        The concentration property label.
    aggregate : db.Compound
        The compound as a linked object.
    properties : db.Collection
        The property collection.
    structures : db.Collection
        The structure collection.
    model : db.Model
        The electronic structure model.

    Returns
    -------
    float
        The concentrations according to property label and electronic structure model. Returns an empty list if no
        property is present.
    """
    centroid = db.Structure(aggregate.get_centroid(), structures)
    property_list = centroid.query_properties(label, model, properties)
    concentrations: List[float] = list()
    for prop_id in property_list:
        prop = db.NumberProperty(prop_id, properties)
        concentrations.append(prop.get_data())
    return concentrations


def query_concentrations(label: str, aggregate: Union[db.Compound, db.Flask], properties: db.Collection,
                         structures: db.Collection) -> List[float]:
    """
    Query all concentrations with a given label independent of the electronic structure model.

    Parameters
    ----------
    label : str
        The concentration property label.
    aggregate : db.Compound
        The compound as a linked object.
    properties : db.Collection
        The property collection.
    structures : db.Collection
        The structure collection.

    Returns
    -------
    float
        The concentrations according to property label and electronic structure model. Returns an empty list if no
        property is present.
    """
    centroid = db.Structure(aggregate.get_centroid(), structures)
    property_list = centroid.get_properties(label)
    concentrations: List[float] = list()
    for prop_id in property_list:
        prop = db.NumberProperty(prop_id, properties)
        concentrations.append(prop.get_data())
    return concentrations


def query_reaction_fluxes_with_model(label_post_fix: str, reaction: db.Reaction, compounds: db.Collection,
                                     flasks: db.Collection, structures: db.Collection, properties: db.Collection,
                                     model: db.Model) -> List[float]:
    """
    Query all concentration fluxes with a given label and electronic structure model.

    Parameters
    ----------
    label_post_fix : str
        The flux label post-fix.
    reaction : db.Reaction
        The reaction.
    compounds : db.Collection
        The compound collection.
    flasks : db.Collection.
        The flask collection.
    properties : db.Collection
        The property collection.
    structures : db.Collection
        The structure collection.
    model : db.Model
        The electronic structure model.

    Returns
    -------
    float
        The concentrations according to property label and electronic structure model. Returns an empty list if no
        property is present.
    """
    label = reaction.id().string() + label_post_fix
    a_id = reaction.get_reactants(db.Side.LHS)[0][0]
    a_type = reaction.get_reactant_types(db.Side.LHS)[0][0]
    aggregate = get_compound_or_flask(a_id, a_type, compounds, flasks)
    return query_concentrations_with_model(label, aggregate, properties, structures, model)


def query_reaction_fluxes(label_post_fix: str, reaction: db.Reaction, compounds: db.Collection,
                          flasks: db.Collection, structures: db.Collection, properties: db.Collection) -> List[float]:
    """
    Query all concentration fluxes with a given label independent of the electronic structure model.

    Parameters
    ----------
    label_post_fix : str
        The flux label post-fix.
    reaction : db.Reaction
        The reaction.
    compounds : db.Collection
        The compound collection.
    flasks : db.Collection.
        The flask collection.
    properties : db.Collection
        The property collection.
    structures : db.Collection
        The structure collection.

    Returns
    -------
    float
        The concentrations according to property label and electronic structure model. Returns an empty list if no
        property is present.
    """
    label = reaction.id().string() + label_post_fix
    a_id = reaction.get_reactants(db.Side.LHS)[0][0]
    a_type = reaction.get_reactant_types(db.Side.LHS)[0][0]
    aggregate = get_compound_or_flask(a_id, a_type, compounds, flasks)
    return query_concentrations(label, aggregate, properties, structures)
