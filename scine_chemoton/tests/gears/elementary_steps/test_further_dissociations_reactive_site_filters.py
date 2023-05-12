#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
import json
import unittest
from itertools import combinations

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application tests imports
from ... import test_database_setup as db_setup
from ...resources import resources_root_path

# Local application imports
from ....gears.elementary_steps.trial_generator.fast_dissociations import (
    FurtherExplorationFilter,
    AllBarrierLessDissociationsFilter,
    ReactionCoordinateMaxDissociationEnergyFilter
)


class FurtherDissociationsReactiveSiteFiltersTests(unittest.TestCase):

    def setUp(self) -> None:
        manager = db_setup.get_clean_db("chemoton_test_further_dissociations_filters")
        structures = manager.get_collection("structures")
        rr = resources_root_path()
        educt = json.load(open(os.path.join(rr, "proline_acid_propanal_product.json"), "r"))
        structure = db.Structure(db.ID(), structures)
        structure.create(os.path.join(rr, "proline_acid_propanal_product.xyz"), 0, 1)
        model = db.Model("FAKE", "", "")
        structure.set_model(model)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", educt["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", educt["masm_idx_map"])
        compound = db.Compound(db.ID(), manager.get_collection("compounds"))
        compound.create([structure.id()])
        structure.set_aggregate(compound.id())
        p1 = db.Structure(db.ID(), structures)
        p1.create(os.path.join(rr, "proline_acid.xyz"), 0, 1)
        p2 = db.Structure(db.ID(), structures)
        p2.create(os.path.join(rr, "propanal.xyz"), 0, 1)
        self.structure = structure
        self.compound = compound
        self.products = [p1.id(), p2.id()]
        self.job = db.Job('scine_dissociation_cut')
        self.manager = manager
        n_atoms = len(structure.get_atoms())
        self.test_atoms = list(range(n_atoms))
        self.test_pairs = list(combinations(range(n_atoms), 2))
        self.test_coordinate = [[(0, 1), (1, 2)]]
        self.dissociations = [0, 1, 1, 2]

    def tearDown(self) -> None:
        self.manager.wipe()

    def _empty_input(self, f: FurtherExplorationFilter) -> None:
        assert [] == f.filter_atoms([self.structure], [])
        assert [] == f.filter_atom_pairs([self.structure], [])
        assert [] == f.filter_reaction_coordinates([self.structure], [])

    def _no_filter_below_reaction_coordinate(self, f: FurtherExplorationFilter) -> None:
        allowed_atoms = f.filter_atoms([self.structure], self.test_atoms)
        assert len(allowed_atoms) == len(self.test_atoms)

        allowed_pairs = f.filter_atom_pairs([self.structure], self.test_pairs)
        assert len(allowed_pairs) == len(self.test_pairs)

    def test_standard_filter(self):
        f = FurtherExplorationFilter()
        f.initialize_collections(self.manager)

        self._empty_input(f)
        self._no_filter_below_reaction_coordinate(f)

        allowed_coordinates = f.filter_reaction_coordinates(self.structure, self.test_coordinate)
        assert len(allowed_coordinates) == len(self.test_coordinate)

    def test_barrierless_filter(self):
        f = AllBarrierLessDissociationsFilter(self.structure.get_model(), self.job.order)
        f.initialize_collections(self.manager)

        self._empty_input(f)
        self._no_filter_below_reaction_coordinate(f)

        # no calculations so this will be outruled
        allowed_coordinates = f.filter_reaction_coordinates(self.structure, self.test_coordinate)
        assert not allowed_coordinates

        calculation = db.Calculation(db.ID(), self.manager.get_collection("calculations"))
        calculation.create(self.structure.get_model(), self.job, [self.structure.id()])
        calculation.set_settings(utils.ValueCollection({'dissociations': self.dissociations}))
        step = db.ElementaryStep.make([self.structure.id()], self.products,
                                      self.manager.get_collection("elementary_steps"))
        reaction = db.Reaction(db.ID(), self.manager.get_collection("reactions"))
        reaction.create([self.compound.id()], [db.ID()], [db.CompoundOrFlask.COMPOUND], [db.CompoundOrFlask.COMPOUND])
        reaction.add_elementary_step(step.id())
        step.set_reaction(reaction.id())
        results = db.Results()
        results.add_elementary_step(step.id())
        calculation.set_results(results)
        calculation.set_status(db.Status.COMPLETE)
        self.structure.add_calculation(calculation.job.order, calculation.id())

        # only works for barrierless
        allowed_coordinates = f.filter_reaction_coordinates(self.structure, self.test_coordinate)
        assert not allowed_coordinates

        step.set_type(db.ElementaryStepType.BARRIERLESS)
        allowed_coordinates = f.filter_reaction_coordinates(self.structure, self.test_coordinate)
        assert len(allowed_coordinates) == len(self.test_coordinate)

        # respects disable
        step.disable_exploration()
        allowed_coordinates = f.filter_reaction_coordinates(self.structure, self.test_coordinate)
        assert not allowed_coordinates

        # wrong coordinate is still outruled
        allowed_coordinates = f.filter_reaction_coordinates(self.structure, [[(0, 1), (1, 5)]])
        assert not allowed_coordinates

    def test_reaction_coordinate_max_dissociation_energy_filter(self):
        energy_type = 'gibbs_free_energy'
        f = ReactionCoordinateMaxDissociationEnergyFilter(100.0, energy_type, self.structure.get_model(),
                                                          self.job.order)
        f.initialize_collections(self.manager)

        self._empty_input(f)
        self._no_filter_below_reaction_coordinate(f)

        # no calculations so this will be outruled
        allowed_coordinates = f.filter_reaction_coordinates(self.structure, self.test_coordinate)
        assert not allowed_coordinates

        calculation = db.Calculation(db.ID(), self.manager.get_collection("calculations"))
        calculation.create(self.structure.get_model(), self.job, [self.structure.id()])
        calculation.set_settings(utils.ValueCollection({'dissociations': self.dissociations}))
        self.structure.add_calculation(calculation.job.order, calculation.id())
        calculation.set_status(db.Status.COMPLETE)

        # also works without steps
        struc_prop = db.NumberProperty(db.ID(), self.manager.get_collection("properties"))
        struc_prop.create(self.structure.get_model(), energy_type, self.structure.id(), calculation.id(),
                          50.0 * utils.HARTREE_PER_KJPERMOL)
        self.structure.add_property(energy_type, struc_prop.id())

        # too high properties
        p0_prop = db.NumberProperty(db.ID(), self.manager.get_collection("properties"))
        p0_prop.create(self.structure.get_model(), energy_type, self.products[0], calculation.id(),
                       70.0 * utils.HARTREE_PER_KJPERMOL)
        db.Structure(self.products[0], self.manager.get_collection("structures")).add_property(energy_type,
                                                                                               p0_prop.id())
        p1_prop = db.NumberProperty(db.ID(), self.manager.get_collection("properties"))
        p1_prop.create(self.structure.get_model(), energy_type, self.products[1], calculation.id(),
                       90.0 * utils.HARTREE_PER_KJPERMOL)
        db.Structure(self.products[1], self.manager.get_collection("structures")).add_property(energy_type,
                                                                                               p1_prop.id())
        product_prop = db.StringProperty(db.ID(), self.manager.get_collection("properties"))
        product_prop.create(self.structure.get_model(), "dissociated_structures", ",".join(str(p)
                                                                                           for p in self.products))

        results = db.Results()
        results.set_properties([struc_prop.id(), p0_prop.id(), p1_prop.id(), product_prop.id()])
        calculation.set_results(results)

        allowed_coordinates = f.filter_reaction_coordinates(self.structure, self.test_coordinate)
        assert not allowed_coordinates

        # low enough properties
        p0_prop.set_data(60.0 * utils.HARTREE_PER_KJPERMOL)
        p1_prop.set_data(70.0 * utils.HARTREE_PER_KJPERMOL)
        allowed_coordinates = f.filter_reaction_coordinates(self.structure, self.test_coordinate)
        assert len(allowed_coordinates) == len(self.test_coordinate)

        # make properties too high again and introduce new product with barrierless step, but also too high
        p0_prop.set_data(80.0 * utils.HARTREE_PER_KJPERMOL)
        p1_prop.set_data(90.0 * utils.HARTREE_PER_KJPERMOL)
        p3 = db.Structure(db.ID(), self.manager.get_collection("structures"))
        p3.create(os.path.join(resources_root_path(), "propanal.xyz"), 0, 1)
        p3_prop = db.NumberProperty(db.ID(), self.manager.get_collection("properties"))
        p3_prop.create(self.structure.get_model(), energy_type, self.products[1], calculation.id(),
                       80.0 * utils.HARTREE_PER_KJPERMOL)
        p3.add_property(energy_type, p3_prop.id())
        results.add_property(p3_prop.id())

        step = db.ElementaryStep.make([self.structure.id()], [self.products[0], p3.id()],
                                      self.manager.get_collection("elementary_steps"))
        step.set_type(db.ElementaryStepType.BARRIERLESS)
        results.add_elementary_step(step.id())
        calculation.set_results(results)

        allowed_coordinates = f.filter_reaction_coordinates(self.structure, self.test_coordinate)
        assert not allowed_coordinates

        # now make barrierless step low enough
        p3_prop.set_data(50.0 * utils.HARTREE_PER_KJPERMOL)
        allowed_coordinates = f.filter_reaction_coordinates(self.structure, self.test_coordinate)
        assert len(allowed_coordinates) == len(self.test_coordinate)
