#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import load
import os
import random
import unittest

# Third party imports
import numpy as np
import scine_database as db
import scine_molassembler as masm
import scine_utilities as utils
from scine_database import test_database_setup as db_setup

# Local application tests imports
from ..resources import resources_root_path

# Local application imports
from scine_chemoton.steering_wheel.selections.conformers import (
    ClusterSelection,
    CentroidConformerSelection,
    LowestEnergyConformerSelection,
    LowestEnergyConformerPerClusterSelection,
    ClusterCentroidConformerSelection
)
from scine_chemoton.steering_wheel.datastructures import NetworkExpansionResult
from scine_chemoton.utilities import connect_to_db


class ConformerSelectionTests(unittest.TestCase):

    def setUp(self) -> None:
        manager = db_setup.get_clean_db(f"chemoton_{self.__class__.__name__}")
        structures = manager.get_collection("structures")
        properties = manager.get_collection("properties")
        self.structures = structures
        rr = resources_root_path()
        path = os.path.join(rr, "proline_acid_propanal_product.xyz")

        self.model = db_setup.get_fake_model()
        self.structure = db.Structure(db.ID(), structures)
        self.structure.create(path, 0, 1, self.model, db.Label.MINIMUM_OPTIMIZED)

        compounds = manager.get_collection("compounds")
        compound = db.Compound(db.ID(), compounds)
        compound.create([self.structure.id()])
        self.structure.set_aggregate(compound.id())

        graph = load(open(os.path.join(rr, "proline_acid_propanal_product.json"), "r"))
        self.structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
        self.structure.set_graph("masm_idx_map", graph["masm_idx_map"])
        self.structure.set_graph("masm_decision_list", graph["masm_decision_list"])
        db_setup.add_random_energy(self.structure, (150, 150.1), properties)

        # generate conformers
        atoms = self.structure.get_atoms()
        bond_orders = utils.BondDetector.detect_bonds(atoms)
        results = masm.interpret.molecules(atoms, bond_orders, masm.interpret.BondDiscretization.Binary)
        if len(results.molecules) > 1:
            raise RuntimeError('Too many molecules, expected only one.')

        masm.Options.Thermalization.disable()
        alignment = masm.BondStereopermutator.Alignment.BetweenEclipsedAndStaggered
        generator = masm.DirectedConformerGenerator(results.molecules[0], alignment)

        lowest_energy_conformer = 75

        model = self.model
        compound_id = compound.id()

        def store(decision_list, conformation):
            """Enumeration callback storing the conformation in the DB"""
            count = len(compound.get_structures())
            # Relabel decision_lists into binned lists
            bin_list = generator.bin_midpoint_integers(decision_list)
            atoms.positions = conformation
            new_structure = db.Structure(db.ID(), structures)
            new_id = new_structure.create(atoms, 0, 1, model, db.Label.MINIMUM_OPTIMIZED)
            new_structure.set_graph("masm_cbor_graph", graph["masm_cbor_graph"])
            new_structure.set_graph("masm_decision_list", "-".join([str(i) for i in bin_list]))
            compound.add_structure(new_id)
            new_structure.set_aggregate(compound_id)
            # expected number of confs ~150, so we pick something in the middle to be the lowest
            energy = abs(lowest_energy_conformer - count)
            db_setup.add_random_energy(new_structure, (energy, energy + 0.1), properties)

        enumeration_settings = generator.EnumerationSettings()
        enumeration_settings.dihedral_retries = 10
        enumeration_settings.fitting = masm.BondStereopermutator.FittingMode.Nearest
        generator.enumerate(callback=store, seed=42, settings=enumeration_settings)

        self.lowest_energy_conformer = lowest_energy_conformer
        self.credentials = manager.get_credentials()
        self.manager = connect_to_db(self.credentials)
        compounds = self.manager.get_collection("compounds")
        self.compound = compound
        self.compound.link(compounds)

    def tearDown(self) -> None:
        self.manager.wipe()

    def _determine_structure_index_in_compound(self, sid: db.ID) -> int:
        structure_indices = np.array(self.compound.get_structures())
        index = np.where(structure_indices == sid, structure_indices, 0).nonzero()
        assert index
        assert len(index[0]) == 1
        return index[0][0]  # pylint: disable=E1136

    def _build_distance_matrix(self) -> np.ndarray:
        structures = self.compound.get_structures(self.manager)
        return ClusterSelection._distance_matrix(structures)

    def test_empty_init_fails(self):
        for cls in [
            ClusterCentroidConformerSelection,
            LowestEnergyConformerPerClusterSelection,
        ]:
            with self.assertRaises(ValueError) as context:
                _ = cls(self.model)
            self.assertTrue('Must specify' in str(context.exception))

            with self.assertRaises(ValueError) as context:
                _ = cls(self.model, n_clusters=0, cluster_rmsd_cutoff=0.0)
            self.assertTrue('Must specify' in str(context.exception))

    def test_double_init_fails(self):
        for cls in [
            ClusterCentroidConformerSelection,
            LowestEnergyConformerPerClusterSelection,
        ]:
            with self.assertRaises(ValueError) as context:
                _ = cls(self.model, n_clusters=5, cluster_rmsd_cutoff=2.0)
            self.assertTrue('Cannot specify' in str(context.exception))

    def test_negative_fails(self):
        for cls in [
            ClusterCentroidConformerSelection,
            LowestEnergyConformerPerClusterSelection,
        ]:
            with self.assertRaises(ValueError) as context:
                _ = cls(self.model, n_clusters=-5)
            self.assertTrue('must be positive' in str(context.exception))

            with self.assertRaises(ValueError) as context:
                _ = cls(self.model, cluster_rmsd_cutoff=-2.0)
            self.assertTrue('must be positive' in str(context.exception))

    def test_result_access_works(self):
        fake_result = NetworkExpansionResult()
        for cls in [
            CentroidConformerSelection,
            LowestEnergyConformerSelection,
        ]:
            inst = cls(self.model)
            inst._step_result = fake_result
            step_result = inst.get_step_result()
            inst.set_step_result(step_result)
        for cls in [
            LowestEnergyConformerPerClusterSelection,
            ClusterCentroidConformerSelection
        ]:
            inst = cls(self.model, n_clusters=4)
            inst._step_result = fake_result
            step_result = inst.get_step_result()
            inst.set_step_result(step_result)

    def test_centroid(self):
        sele = CentroidConformerSelection(self.model)
        step_result = NetworkExpansionResult(compounds=[self.compound.id()])
        result = sele(self.credentials, step_result)
        assert len(result.structures) == 1
        assert result.structures[0] == self.structure.id()  # pylint: disable=unsubscriptable-object

    def test_lowest(self):
        sele = LowestEnergyConformerSelection(self.model)
        step_result = NetworkExpansionResult(compounds=[self.compound.id()])
        result = sele(self.credentials, step_result)
        assert len(result.structures) == 1
        assert str(
            result.structures[0]) == str(  # pylint: disable=unsubscriptable-object
            self.compound.get_structures()[
                self.lowest_energy_conformer])

    def test_cluster_centroid(self):
        sele = ClusterCentroidConformerSelection(self.model, n_clusters=1)
        self.impl_test_cluster_centroid(sele)
        sele = ClusterCentroidConformerSelection(self.model, n_clusters=1, cluster_rmsd_cutoff=0.0)
        self.impl_test_cluster_centroid(sele)

    def impl_test_cluster_centroid(self, sele: ClusterCentroidConformerSelection):
        step_result = NetworkExpansionResult(compounds=[self.compound.id()])
        result = sele(self.credentials, step_result)
        assert len(result.structures) == 1
        dist_matrix = self._build_distance_matrix()
        centroid_id = result.structures[0]  # pylint: disable=unsubscriptable-object
        centroid_index = self._determine_structure_index_in_compound(centroid_id)
        n = len(self.compound.get_structures())
        centroid_rmsd_sum = sum(dist_matrix[centroid_index][i] for i in range(n))
        for j in range(n):
            if j == centroid_index:
                continue
            val = sum(dist_matrix[j][i] for i in range(n))
            assert val > centroid_rmsd_sum

    def test_cluster_lowest(self):
        sele = LowestEnergyConformerPerClusterSelection(self.model, n_clusters=1)
        self.impl_test_cluster_lowest(sele)
        sele = LowestEnergyConformerPerClusterSelection(self.model, n_clusters=1, cluster_rmsd_cutoff=0.0)
        self.impl_test_cluster_lowest(sele)

    def impl_test_cluster_lowest(self, sele: LowestEnergyConformerPerClusterSelection):
        step_result = NetworkExpansionResult(compounds=[self.compound.id()])
        result = sele(self.credentials, step_result)
        assert len(result.structures) == 1
        lowest_id = result.structures[0]  # pylint: disable=unsubscriptable-object
        lowest_index = self._determine_structure_index_in_compound(lowest_id)
        assert lowest_index == self.lowest_energy_conformer

    def test_correct_cluster_num(self):
        step_result = NetworkExpansionResult(compounds=[self.compound.id()])
        random_n_cluster = random.randint(1, len(self.compound.get_structures()) - 1)

        sele = ClusterCentroidConformerSelection(self.model, n_clusters=random_n_cluster)
        result = sele(self.credentials, step_result)
        assert len(result.structures) == random_n_cluster

        sele = LowestEnergyConformerPerClusterSelection(self.model, n_clusters=random_n_cluster)
        result = sele(self.credentials, step_result)
        assert len(result.structures) == random_n_cluster

        sele = ClusterCentroidConformerSelection(self.model, n_clusters=random_n_cluster, cluster_rmsd_cutoff=0.0)
        result = sele(self.credentials, step_result)
        assert len(result.structures) == random_n_cluster

        sele = LowestEnergyConformerPerClusterSelection(
            self.model, n_clusters=random_n_cluster, cluster_rmsd_cutoff=0.0)
        result = sele(self.credentials, step_result)
        assert len(result.structures) == random_n_cluster

    def test_correct_rmsd_distance(self):
        random_rmsd = random.random() * 5.0 + 0.1  # don't want zero
        kwargs = {"cluster_rmsd_cutoff": random_rmsd}
        self.impl_test_correct_rmsd_distance(kwargs)
        kwargs["n_clusters"] = 0
        self.impl_test_correct_rmsd_distance(kwargs)

    def impl_test_correct_rmsd_distance(self, kwargs):
        step_result = NetworkExpansionResult(compounds=[self.compound.id()])
        dist_matrix = self._build_distance_matrix()
        sele = ClusterCentroidConformerSelection(self.model, **kwargs)
        result = sele(self.credentials, step_result)
        assert result.structures
        for i, ss_i in enumerate(result.structures):
            index_i = self._determine_structure_index_in_compound(ss_i)
            for j, ss_j in enumerate(result.structures):
                if i <= j:
                    continue
                index_j = self._determine_structure_index_in_compound(ss_j)
                assert dist_matrix[index_i][index_j] >= kwargs["cluster_rmsd_cutoff"]

        sele = LowestEnergyConformerPerClusterSelection(self.model, **kwargs)
        result = sele(self.credentials, step_result)
        assert result.structures
        for i, ss_i in enumerate(result.structures):
            index_i = self._determine_structure_index_in_compound(ss_i)
            for j, ss_j in enumerate(result.structures):
                if i <= j:
                    continue
                index_j = self._determine_structure_index_in_compound(ss_j)
                assert dist_matrix[index_i][index_j] >= kwargs["cluster_rmsd_cutoff"]
