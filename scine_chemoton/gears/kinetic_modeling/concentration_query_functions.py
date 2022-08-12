#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Union

# Third party imports
import scine_database as db


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
