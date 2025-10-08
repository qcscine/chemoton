#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from inspect import currentframe
from typing import List
import os
import warnings

# Third party imports
import scine_utilities as utils
import scine_database as db
import scine_molassembler as masm
import numpy as np
import pytest
from pymatgen.core.surface import Slab
from scine_database.test_database_setup import get_clean_db

# Local application tests imports
from ..resources import resources_root_path

# Local application imports
from ...utilities.reactive_complexes.adsorption import (
    Adsorbate,
    AdsorptionResult,
    AdsorptionGenerator,
)
from ...utilities.insert_initial_structure import insert_surface_from_cif_file, insert_initial_structure
from ...utilities.surfaces.pymatgen_interface import PmgInterface


def no_api_key() -> bool:
    potential_rc_file = os.path.join(os.path.expanduser("~"), ".pmgrc.yaml")
    key = "PMG_MAPI_KEY"
    if os.environ.get(key) is not None or (
            os.path.exists(potential_rc_file) and key in open(potential_rc_file).read()):
        return False
    return True


def check_result_consistency(results: List[AdsorptionResult], n_mols: int) -> None:
    for r in results:
        slab = Slab.from_dict(r.slab_dict)
        assert slab is not None
        slab_ps = PmgInterface.to_periodic_system(slab, r.surface_atom_indices)
        assert slab_ps.is_approx(r.ps, 1e-6)
        data = r.ps.get_data_for_molassembler_interpretation()
        mol_result = masm.interpret.molecules(*data, masm.interpret.BondDiscretization.Binary)
        assert len(mol_result.molecules) == n_mols


def test_adsorbate():
    manager = get_clean_db("chemoton_" + currentframe().f_code.co_name)
    rr = resources_root_path()
    model = db.Model("FAKE", "FAKE", "F-AKE")
    water_path = os.path.join(rr, "water.xyz")
    water, _ = insert_initial_structure(manager, water_path, 0, 1, model)
    water_ac = water.get_atoms()
    adsorbate = Adsorbate(water, [0])
    assert adsorbate.atoms == water_ac
    assert len(adsorbate.reactive_atoms) == 1
    assert len(adsorbate.directions)
    assert len(adsorbate.vdw_values)
    assert adsorbate.vdw_avg > 0
    assert adsorbate.vdw_avg_reactive > 0
    assert adsorbate.sites is None
    assert abs(adsorbate.get_max_extension() - np.linalg.norm(water_ac.positions[0] - water_ac.positions[1])) < 1e-6

    adsorbate = Adsorbate(water)
    assert len(adsorbate.reactive_atoms) == len(water_ac)
    assert len(adsorbate.directions)
    assert len(adsorbate.vdw_values)
    assert adsorbate.vdw_avg > 0
    assert adsorbate.vdw_avg_reactive > 0
    assert adsorbate.sites is None
    assert abs(adsorbate.get_max_extension() - np.linalg.norm(water_ac.positions[0] - water_ac.positions[1])) < 1e-6


def test_adsorption_complete_multiple_adsorbates():
    manager = get_clean_db("chemoton_" + currentframe().f_code.co_name)
    rr = resources_root_path()
    model = db.Model("FAKE", "FAKE", "F-AKE")
    water_path = os.path.join(rr, "water.xyz")
    water, _ = insert_initial_structure(manager, water_path, 0, 1, model)
    cu_path = os.path.join(rr, "cu.cif")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        cu, _ = insert_surface_from_cif_file(manager, cu_path, "111", 0, 1, model,
                                             slab_settings={'min_slab_layers': 3.0})[0]
    cu_pbc = utils.PeriodicBoundaries(cu.get_model().periodic_boundaries)
    cu_pbc_3_by_3 = cu_pbc * [3, 3, 1]
    assert len(cu.get_atoms()) == 3  # 1 * 3 layers
    generator = AdsorptionGenerator()
    generator.properties = manager.get_collection("properties")
    generator.options.wanted_coverage = 0.33
    generator.options.maximum_extension = 1
    generator.options.check_size = True
    generator.options.multiple_molecules = False
    generator.options.rotamers = False

    with pytest.raises(RuntimeError) as e_info:
        _ = generator.generate_reactive_complexes(cu, water)
    generator.options.maximum_extension = 4
    assert "Could not get right coverage" in str(e_info)

    with pytest.raises(RuntimeError) as e_info:
        _ = generator.generate_reactive_complexes(cu, water)
    assert "Could not get right coverage" in str(e_info)

    generator.options.multiple_molecules = True
    results = generator.generate_reactive_complexes(cu, water)
    assert results
    assert len(results) == 16  # 4 sites on Cu(111) and 4 reactive sites in water (Oxygen has 2)
    assert all(len(r.atoms) == 36 for r in results)  # 9*3 Cu atoms + 3 waters = 36 atoms in each result
    assert all(r.slab_extension == [3, 3, 1] for r in results)
    assert all(r.pbc == cu_pbc_3_by_3 for r in results)
    check_result_consistency(results, 4)


@pytest.mark.skipif(no_api_key(), reason="Test requires a MaterialsProject API key; set 'PMG_MAPI_KEY'")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_adsorption_complete_full_coverage():
    manager = get_clean_db("chemoton_" + currentframe().f_code.co_name)
    rr = resources_root_path()
    model = db.Model("FAKE", "FAKE", "F-AKE")
    water_path = os.path.join(rr, "water.xyz")
    water, _ = insert_initial_structure(manager, water_path, 0, 1, model)
    cu_path = os.path.join(rr, "cu.cif")
    cu, _ = insert_surface_from_cif_file(manager, cu_path, "111", 0, 1, model,
                                         slab_settings={'min_slab_layers': 3.0})[0]
    cu_pbc = utils.PeriodicBoundaries(cu.get_model().periodic_boundaries)
    assert len(cu.get_atoms()) == 3  # 1 * 3 layers
    generator = AdsorptionGenerator()
    generator.properties = manager.get_collection("properties")
    generator.options.wanted_coverage = 1.0
    generator.options.maximum_extension = 10
    generator.options.check_size = True
    generator.options.multiple_molecules = False
    results = generator.generate_reactive_complexes(cu, water)
    assert results
    assert len(results) == 16  # 4 sites on Cu(111) and 3 reactive sites in water with 2 directions for O
    assert all(len(r.atoms) == 6 for r in results)  # 1*3 Cu atoms + 1 water = 6 atoms in each result
    assert all(r.slab_extension == [1, 1, 1] for r in results)
    assert all(r.pbc == cu_pbc for r in results)
    check_result_consistency(results, 2)


@pytest.mark.skipif(no_api_key(), reason="Test requires a MaterialsProject API key; set 'PMG_MAPI_KEY'")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_adsorption_complete_full_coverage_larger_surface():
    manager = get_clean_db("chemoton_" + currentframe().f_code.co_name)
    rr = resources_root_path()
    model = db.Model("FAKE", "FAKE", "F-AKE")
    water_path = os.path.join(rr, "water.xyz")
    water, _ = insert_initial_structure(manager, water_path, 0, 1, model)
    cu_path = os.path.join(rr, "cu.cif")
    cu, _ = insert_surface_from_cif_file(manager, cu_path, "111", 0, 1, model, extension=4,
                                         slab_settings={'min_slab_layers': 3.0})[0]
    cu_pbc = utils.PeriodicBoundaries(cu.get_model().periodic_boundaries)
    assert len(cu.get_atoms()) == 48  # 4 * 4 * 3 layers
    generator = AdsorptionGenerator()
    generator.properties = manager.get_collection("properties")
    generator.options.wanted_coverage = 1.0
    generator.options.maximum_extension = 10
    generator.options.check_size = True
    generator.options.multiple_molecules = False
    with pytest.raises(RuntimeError) as e_info:
        _ = generator.generate_reactive_complexes(cu, water)
    assert "Cannot achieve coverage" in str(e_info)

    generator.options.multiple_molecules = True
    generator.options.rotamers = True
    results = generator.generate_reactive_complexes(cu, water)
    assert results
    assert len(results) == 16  # 4 sites on Cu(111) and 4 reactive sites in water (Oxygen has 2)
    assert all(len(r.atoms) == 96 for r in results)  # 4*4*3 Cu atoms + 16 water = 96 atoms in each result
    assert all(r.slab_extension == [1, 1, 1] for r in results)
    assert all(r.pbc == cu_pbc for r in results)
    check_result_consistency(results, 17)


@pytest.mark.skipif(no_api_key(), reason="Test requires a MaterialsProject API key; set 'PMG_MAPI_KEY'")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_adsorption_complete_single_adsorbate():
    manager = get_clean_db("chemoton_" + currentframe().f_code.co_name)
    rr = resources_root_path()
    model = db.Model("FAKE", "FAKE", "F-AKE")
    water_path = os.path.join(rr, "water.xyz")
    water, _ = insert_initial_structure(manager, water_path, 0, 1, model)
    cu_path = os.path.join(rr, "cu.cif")
    cu, _ = insert_surface_from_cif_file(manager, cu_path, "111", 0, 1, model,
                                         slab_settings={'min_slab_layers': 3.0})[0]
    cu_pbc = utils.PeriodicBoundaries(cu.get_model().periodic_boundaries)
    cu_pbc_2_by_2 = cu_pbc * [2, 2, 1]
    assert len(cu.get_atoms()) == 3  # 1 * 3 layers
    generator = AdsorptionGenerator()
    generator.properties = manager.get_collection("properties")
    generator.options.wanted_coverage = 0.0
    generator.options.maximum_extension = 10
    generator.options.check_size = True
    generator.options.multiple_molecules = False
    generator.options.extension = [2, 2, 1]  # activate single mol mode
    generator.options.rotamers = False
    results = generator.generate_reactive_complexes(cu, water)
    assert results
    assert len(results) == 16  # 4 sites on Cu(111) and 4 reactive sites in water (Oxygen has 2)
    assert all(len(r.atoms) == 15 for r in results)  # 2*2*3 Cu atoms + 1 water = 15 atoms in each result
    assert all(r.slab_extension == [2, 2, 1] for r in results)
    assert all(r.pbc == cu_pbc_2_by_2 for r in results)
    check_result_consistency(results, 2)
