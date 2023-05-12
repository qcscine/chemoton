#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Third party imports
import scine_database as db
from warnings import warn


def insert_concentration_for_compound(
        database: db.Manager,
        value: float,
        model: db.Model,
        compound_id: db.ID,
        replace_old: bool = True,
        label: str = "start_concentration"
):
    compounds = database.get_collection("compounds")
    compound = db.Compound(compound_id, compounds)
    centroid_structure = compound.get_centroid()
    insert_concentration_for_structure(database, value, model, centroid_structure, replace_old, label)


def insert_concentration_for_structure(
        database: db.Manager,
        value: float,
        model: db.Model,
        structure_id: db.ID,
        replace_old: bool = True,
        label: str = "start_concentration"):
    concentration_options = ["start_concentration", "max_concentration", "final_concentration", "concentration_flux",
                             "manual_activation"]
    if label not in concentration_options and "_concentration_flux" not in label:
        warn(f"Your concentration label is not within the suggested labels {str(concentration_options)}."
             f"This may lead to problems recognizing the concentrations in the individual gears!")

    properties = database.get_collection("properties")
    structures = database.get_collection("structures")

    structure = db.Structure(structure_id, structures)

    if structure.has_property(label) and replace_old:
        concentration_prop = db.NumberProperty(structure.get_properties(label)[-1], properties)
        concentration_prop.set_data(value)
        return

    concentration_prop = db.NumberProperty.make(label, model, value, properties)
    concentration_prop.set_structure(structure.id())
    structure.add_property(label, concentration_prop.id())
