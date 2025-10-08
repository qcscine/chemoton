#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""


# Standard library imports
from inspect import currentframe
import os
import pytest

# Third party imports
import scine_database as db
import scine_utilities as utils
from scine_database.test_database_setup import get_clean_db

# Local application tests imports
from ..resources import resources_root_path

# Local application imports
from ...utilities.insert_initial_structure import insert_surface_from_materials_project, insert_surface_from_cif_file


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_insert_surface_from_cif():
    manager = get_clean_db("chemoton_" + currentframe().f_code.co_name)
    rr = resources_root_path()
    model = db.Model("FAKE", "FAKE", "F-AKE")
    cu_path = os.path.join(rr, "cu.cif")
    inputs = insert_surface_from_cif_file(manager, cu_path, "111", 0, 1, model,
                                          db.Label.SURFACE_GUESS,
                                          db.Job('scine_geometry_optimization'),
                                          slab_settings={'min_slab_layers': 3.0}, extension=4)
    assert len(inputs) == 1
    struct, calc = inputs[0]
    assert struct.get_charge() == 0
    assert struct.get_multiplicity() == 1
    assert struct.get_label() == db.Label.SURFACE_GUESS
    assert struct.get_atoms().size() == 48  # 4 * 4 * 3
    assert struct.get_model().method == model.method
    pbc_string = struct.get_model().periodic_boundaries
    assert pbc_string and pbc_string.lower() != "none"
    pbc = utils.PeriodicBoundaries(pbc_string)
    assert pbc is not None
    assert calc.get_job().order == "scine_geometry_optimization"
    assert calc.get_model().method_family == model.method_family
    assert calc.get_status() == db.Status.HOLD
    assert calc.get_structures()[0] == struct.get_id()
    assert not calc.get_settings()

    # Cleaning
    manager.wipe()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_insert_surface_from_cif_max_index():
    manager = get_clean_db("chemoton_" + currentframe().f_code.co_name)
    rr = resources_root_path()
    model = db.Model("FAKE", "FAKE", "F-AKE")
    cu_path = os.path.join(rr, "cu.cif")
    inputs = insert_surface_from_cif_file(manager, cu_path, 1, 0, 1, model,
                                          db.Label.SURFACE_GUESS,
                                          db.Job('scine_geometry_optimization'),
                                          slab_settings={'min_slab_layers': 6.0}, extension=2)
    assert len(inputs) == 4
    for struct, calc in inputs:
        assert struct.get_charge() == 0
        assert struct.get_multiplicity() == 1
        assert struct.get_label() == db.Label.SURFACE_GUESS
        assert struct.get_atoms().size() >= 24  # 2 * 2 * 6
        assert struct.get_model().method == model.method
        pbc_string = struct.get_model().periodic_boundaries
        assert pbc_string and pbc_string.lower() != "none"
        pbc = utils.PeriodicBoundaries(pbc_string)
        assert pbc is not None
        assert calc.get_job().order == "scine_geometry_optimization"
        assert calc.get_model().method_family == model.method_family
        assert calc.get_status() == db.Status.HOLD
        assert calc.get_structures()[0] == struct.get_id()
        assert not calc.get_settings()

    # Cleaning
    manager.wipe()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_insert_surface_from_materialsproject():
    manager = get_clean_db("chemoton_" + currentframe().f_code.co_name)
    model = db.Model("FAKE", "FAKE", "F-AKE")
    inputs = insert_surface_from_materials_project(manager, "Cu", "111", 0, 1, model,
                                                   db.Label.SURFACE_GUESS,
                                                   db.Job('scine_geometry_optimization'),
                                                   slab_settings={'min_slab_layers': 3.0}, extension=4)
    assert len(inputs) == 1
    struct, calc = inputs[0]
    assert struct.get_charge() == 0
    assert struct.get_multiplicity() == 1
    assert struct.get_label() == db.Label.SURFACE_GUESS
    assert struct.get_atoms().size() == 48  # 4 * 4 * 3
    assert struct.get_model().method == model.method
    pbc_string = struct.get_model().periodic_boundaries
    assert pbc_string and pbc_string.lower() != "none"
    pbc = utils.PeriodicBoundaries(pbc_string)
    assert pbc is not None
    assert calc.get_job().order == "scine_geometry_optimization"
    assert calc.get_model().method_family == model.method_family
    assert calc.get_status() == db.Status.HOLD
    assert calc.get_structures()[0] == struct.get_id()
    assert not calc.get_settings()

    # Cleaning
    manager.wipe()
