#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Third party imports
import scine_database as db


def insert_concentration_for_compound(
        database: db.Manager,
        value: float,
        model: db.Model,
        compound_id: db.ID,
        replace_old: bool = True,
        label: str = "start_concentration"
):
    concentration_options = ["start_concentration", "max_concentration", "final_concentration", "concentration_flux"]
    if label not in concentration_options:
        raise RuntimeError("The concentration label must be label either: 'start_concentration', 'max_concentration',"
                           "'final_concentration', or 'concentration_flux'.")

    properties = database.get_collection("properties")
    compounds = database.get_collection("compounds")

    compound = db.Compound(compound_id, compounds)
    centroid_structure = compound.get_centroid(database)

    if centroid_structure.has_property(label) and replace_old:
        concentration_prop = db.NumberProperty(centroid_structure.get_properties(label)[-1], properties)
        concentration_prop.set_data(value)
        return

    concentration_prop = db.NumberProperty.make(label, model, value, properties)
    concentration_prop.set_structure(centroid_structure.id())
    centroid_structure.add_property(label, concentration_prop.id())
