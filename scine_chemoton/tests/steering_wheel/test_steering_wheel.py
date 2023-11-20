#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
from time import sleep
import pytest
import os
import unittest

# Third party imports
import scine_database as db
import scine_utilities as utils
from scine_database import test_database_setup as db_setup

# Local application tests imports
from ..resources import resources_root_path

# Local application imports
from scine_chemoton.gears.compound import BasicAggregateHousekeeping
from scine_chemoton.gears.scheduler import Scheduler
from scine_chemoton.steering_wheel import SteeringWheel
from scine_chemoton.steering_wheel.datastructures import SelectionResult, NetworkExpansionResult, Status, GearOptions
from scine_chemoton.steering_wheel.selections import SelectionAndArray, AllCompoundsSelection, PredeterminedSelection
from scine_chemoton.steering_wheel.selections.input_selections import FileInputSelection
from scine_chemoton.steering_wheel.selections.conformers import (
    LowestEnergyConformerSelection,
    ClusterCentroidConformerSelection,
)
from scine_chemoton.steering_wheel.selections.organometallic_complexes import CentralMetalSelection
from scine_chemoton.steering_wheel.network_expansions import GiveWholeDatabaseWithModelResult
from scine_chemoton.steering_wheel.network_expansions.basics import SimpleOptimization
from scine_chemoton.steering_wheel.network_expansions.conformers import ConformerCreation
from scine_chemoton.steering_wheel.network_expansions.reactions import Dissociation


def _input_yes(_question: str) -> str:
    return "y"


def _input_no(_question: str) -> str:
    return "n"


class SteeringWheelTests(unittest.TestCase):

    def setUp(self) -> None:
        self.manager = db_setup.get_clean_db(f"chemoton_{self.__class__.__name__}")
        self.model = db_setup.get_fake_model()
        self.credentials = self.manager.get_credentials()
        self.current_wheel = None

    def tearDown(self) -> None:
        self.manager.wipe()
        if self.current_wheel is not None:
            if self.current_wheel.is_running():
                self.current_wheel.terminate(try_save_progress=False)
            self.current_wheel.clear_cache()
            self.current_wheel.join()

    def test_empty_init_works(self):
        wheel = SteeringWheel(self.credentials, [])
        assert wheel is not None

    def test_invalid_protocols_fail(self):
        nosele = [ConformerCreation, LowestEnergyConformerSelection, Dissociation]
        with self.assertRaises(TypeError) as context:
            _ = SteeringWheel(self.credentials, nosele)
        self.assertTrue("The first entry in your exploration scheme is not a selection" in str(context.exception))

        noinput = [LowestEnergyConformerSelection, ConformerCreation, LowestEnergyConformerSelection]
        with self.assertRaises(TypeError) as context:
            _ = SteeringWheel(self.credentials, noinput)
        self.assertTrue("The first entry in your exploration scheme is not a selection that inputs"
                        in str(context.exception))

    def test_double_run_fails(self):
        rr = resources_root_path()
        paths = [os.path.join(rr, name) for name in ["grubbs.xyz", "water.xyz"]]
        inp = FileInputSelection(self.model, [(p, 0, 1) for p in paths])
        whole = GiveWholeDatabaseWithModelResult(self.model, status_cycle_time=100)
        protocol = [inp, whole]
        global_selection = CentralMetalSelection(self.model, "Ru", False)
        wheel = SteeringWheel(self.credentials, protocol, global_selection, callable_input=_input_no)
        self.current_wheel = wheel
        wheel.run(allow_restart=False)  # safer if some other tests failed and didn't clean up
        with self.assertRaises(RuntimeError) as context:
            wheel.run()
        self.assertTrue("Already running the exploration protocol" in str(context.exception))
        wheel.stop()
        wheel.run(allow_restart=True)
        wheel.stop(save_progress=False)
        wheel.start(allow_restart=False)
        wheel.stop(save_progress=False)

    def test_gear_option_additions_work(self):
        agg_gear = BasicAggregateHousekeeping()
        agg_gear.options.cycle_time = 1
        schedule_gear = Scheduler()
        schedule_gear.options.cycle_time = 1
        options = GearOptions([schedule_gear])
        assert len(options) == 1
        options += agg_gear
        assert len(options) == 2

    @pytest.mark.slow  # type: ignore[misc]
    def test_addition_to_scheme(self):
        calculations = self.manager.get_collection("calculations")
        structures = self.manager.get_collection("structures")
        rr = resources_root_path()
        paths = [os.path.join(rr, name) for name in ["grubbs.xyz", "water.xyz"]]
        inp = FileInputSelection(self.model, [(p, 0, 1) for p in paths])
        # set shorter cycles for gears
        agg_gear = BasicAggregateHousekeeping()
        agg_gear.options.cycle_time = 1
        agg_gear.options.model = self.model
        schedule_gear = Scheduler()
        schedule_gear.options.cycle_time = 1
        schedule_gear.options.model = self.model
        options = GearOptions([schedule_gear])
        whole = GiveWholeDatabaseWithModelResult(self.model, status_cycle_time=1, gear_options=options)
        more_options = GearOptions([agg_gear, schedule_gear])
        opt = SimpleOptimization(self.model, status_cycle_time=1, gear_options=more_options)
        wheel = SteeringWheel(self.credentials, [inp], callable_input=_input_yes)
        self.current_wheel = wheel
        wheel.run(allow_restart=False)  # safer if some other tests failed and didn't clean up
        while wheel.is_running():
            sleep(1)
        assert len(wheel.get_status_report().values()) == 1
        assert all(v == Status.FINISHED for v in wheel.get_status_report().values())
        assert structures.count("{}") == len(paths)
        # addition while not running
        assert not wheel.is_running()
        wheel += whole
        assert not wheel.is_running()
        assert len(wheel.get_status_report().values()) == 2
        assert structures.count("{}") == len(paths)  # structures should not have been added again
        # addition while running
        result = SelectionResult()
        result.structures = [s.id() for s in structures.query_structures("{}")]
        pre = PredeterminedSelection(self.model, result)
        with pytest.warns(UserWarning):
            # get warning because restart name changed due to new steps, but we still restart,
            # because we find the old name
            wheel.run(allow_restart=True)
        wheel += [pre, opt]
        # optimization step will never finish without puffins
        sleep(5)
        assert wheel.is_running()
        assert len(wheel.get_status_report().values()) == 4
        assert structures.count("{}") == len(paths)
        count = 0
        while list(wheel.get_status_report().values())[3] != Status.CALCULATING:
            count += 1
            if count > 20:
                assert False
            sleep(1)
        sleep(5)
        # manually turn to finished ones
        sid1 = db.ID()
        sid2 = db.ID()
        s1 = db.Structure(sid1, structures)
        s2 = db.Structure(sid2, structures)
        s1.create(paths[0], 0, 1)
        s2.create(paths[1], 0, 1)
        sids = [sid1, sid2]
        assert calculations.count(dumps({})) == 2
        assert calculations.count(dumps({"status": "complete"})) == 0
        assert calculations.count(dumps({"status": "hold"})) == 0
        assert calculations.count(dumps({"status": "new"})) == 2
        for i, calc in enumerate(calculations.query_calculations(dumps({"status": "new"}))):
            result = db.Results()
            result.structure_ids = [sids[i]]
            calc.set_results(result)
            calc.set_status(db.Status.COMPLETE)
        assert calculations.count(dumps({"status": "complete"})) == 2
        assert calculations.count(dumps({"status": "new"})) == 0
        assert calculations.count(dumps({"status": "hold"})) == 0
        while wheel.is_running():
            sleep(1)
        assert len(wheel.get_status_report().values()) == 4
        assert all(v == Status.FINISHED for v in wheel.get_status_report().values())

    @pytest.mark.slow  # type: ignore[misc]
    def test_global_selection(self):
        rr = resources_root_path()
        paths = [os.path.join(rr, name) for name in ["grubbs.xyz", "water.xyz"]]
        inp = FileInputSelection(self.model, [(p, 0, 1) for p in paths])
        whole = GiveWholeDatabaseWithModelResult(self.model, status_cycle_time=1)
        all_sele = AllCompoundsSelection(self.model)
        protocol = [inp, whole, all_sele]
        metal = "Ru"
        global_selection = CentralMetalSelection(self.model, metal, False)
        wheel = SteeringWheel(self.credentials, protocol, global_selection, global_for_first_selection=False)
        self.current_wheel = wheel
        assert len(wheel.get_status_report()) == len(protocol)
        assert len(wheel.scheme) == len(protocol)
        assert isinstance(wheel.scheme[0], FileInputSelection)
        assert isinstance(wheel.scheme[1], GiveWholeDatabaseWithModelResult)
        assert isinstance(wheel.scheme[2], SelectionAndArray)
        assert wheel.get_results() is None
        wheel.run()
        while wheel.is_running():
            sleep(1)
        assert all(v == Status.FINISHED for v in wheel.get_status_report().values())
        results = wheel.get_results()
        assert results is not None
        assert len(results) == 3
        structures = self.manager.get_collection("structures")
        assert structures.count("{}") == 2
        assert isinstance(results[0], SelectionResult)
        assert isinstance(results[1], NetworkExpansionResult)
        assert isinstance(results[2], SelectionResult)
        assert len(results[0].structures) == 2
        assert len(results[1].structures) == 2
        assert len(results[2].structures) == 0
        compounds = self.manager.get_collection("compounds")
        structures = self.manager.get_collection("structures")
        for structure in structures.query_structures("{}"):
            compound = db.Compound(db.ID(), compounds)
            compound.add_structure(structure.id())
            structure.set_aggregate(compound.id())
        f = results[2].aggregate_filter
        for i, compound in enumerate(compounds.query_compounds("{}")):
            if i == 0:
                assert f.filter(compound)
                struc = db.Structure(compound.get_structures()[0], structures)
                assert struc.get_atoms().elements
                assert utils.ElementType.Ru in struc.get_atoms().elements
            else:
                assert not f.filter(compound)
                struc = db.Structure(compound.get_structures()[0], structures)
                assert struc.get_atoms().elements
                assert utils.ElementType.Ru not in struc.get_atoms().elements

    def test_global_selection_setter(self):
        rr = resources_root_path()
        paths = [os.path.join(rr, name) for name in ["grubbs.xyz", "water.xyz"]]
        inp = FileInputSelection(self.model, [(p, 0, 1) for p in paths])
        # set shorter cycles for gears
        agg_gear = BasicAggregateHousekeeping()
        agg_gear.options.cycle_time = 1
        schedule_gear = Scheduler()
        schedule_gear.options.cycle_time = 1
        options = GearOptions([schedule_gear])
        whole = GiveWholeDatabaseWithModelResult(self.model, status_cycle_time=1, gear_options=options)
        more_options = GearOptions([agg_gear, schedule_gear])
        pre = PredeterminedSelection(self.model, SelectionResult())
        opt = SimpleOptimization(self.model, status_cycle_time=1, gear_options=more_options)
        wheel = SteeringWheel(self.credentials, [inp, whole, pre, opt], callable_input=_input_no)
        self.current_wheel = wheel
        assert len(wheel.scheme) == 4
        assert isinstance(wheel.scheme[0], FileInputSelection)
        assert isinstance(wheel.scheme[1], GiveWholeDatabaseWithModelResult)
        assert isinstance(wheel.scheme[2], PredeterminedSelection)
        assert isinstance(wheel.scheme[3], SimpleOptimization)

        global_selection = ClusterCentroidConformerSelection(self.model, n_clusters=2)
        wheel.set_global_selection(global_selection, global_for_first_selection=False)
        assert len(wheel.scheme) == 4
        # 0
        assert not isinstance(wheel.scheme[0], SelectionAndArray)
        assert isinstance(wheel.scheme[0], FileInputSelection)
        # 1
        assert isinstance(wheel.scheme[1], GiveWholeDatabaseWithModelResult)
        # 2
        assert not isinstance(wheel.scheme[2], PredeterminedSelection)
        assert isinstance(wheel.scheme[2], SelectionAndArray)
        assert any(isinstance(s, PredeterminedSelection) for s in wheel.scheme[2].selections)
        assert any(isinstance(s, ClusterCentroidConformerSelection) for s in wheel.scheme[2].selections)
        # 3
        assert isinstance(wheel.scheme[3], SimpleOptimization)

        metal = utils.ElementType.Ru
        new_global_selection = CentralMetalSelection(self.model, metal, False)
        wheel.set_global_selection(new_global_selection, global_for_first_selection=True)
        assert len(wheel.scheme) == 4
        # 0
        assert not isinstance(wheel.scheme[0], FileInputSelection)
        assert isinstance(wheel.scheme[0], SelectionAndArray)
        assert any(isinstance(s, FileInputSelection) for s in wheel.scheme[0].selections)
        assert any(isinstance(s, CentralMetalSelection) for s in wheel.scheme[0].selections)
        assert not any(isinstance(s, ClusterCentroidConformerSelection) for s in wheel.scheme[0].selections)
        # 1
        assert isinstance(wheel.scheme[1], GiveWholeDatabaseWithModelResult)
        # 2
        assert not isinstance(wheel.scheme[2], PredeterminedSelection)
        assert isinstance(wheel.scheme[2], SelectionAndArray)
        assert any(isinstance(s, SelectionAndArray) for s in wheel.scheme[2].selections)
        assert any(isinstance(s, CentralMetalSelection) for s in wheel.scheme[2].selections)
        for s in wheel.scheme[2].selections:
            if isinstance(s, SelectionAndArray):
                assert any(isinstance(si, PredeterminedSelection) for si in s.selections)
                assert any(isinstance(si, ClusterCentroidConformerSelection) for si in s.selections)
        # 3
        assert isinstance(wheel.scheme[3], SimpleOptimization)

    @pytest.mark.slow  # type: ignore[misc]
    def test_large_results(self):
        structures = self.manager.get_collection("structures")
        rr = resources_root_path()
        # 10_000 already caused problems earlier, since this is OS dependent, we use 20_000
        paths = [os.path.join(rr, "water.xyz") for _ in range(20_000)]  # ensures large results
        inp = FileInputSelection(self.model, [(p, 0, 1) for p in paths])
        # set shorter cycles for gear
        schedule_gear = Scheduler()
        schedule_gear.options.cycle_time = 1
        schedule_gear.options.model = self.model
        options = GearOptions([schedule_gear])
        whole = GiveWholeDatabaseWithModelResult(self.model, status_cycle_time=1, gear_options=options)
        all_compounds = AllCompoundsSelection(self.model)
        wheel = SteeringWheel(self.credentials, [inp, whole, all_compounds], callable_input=_input_no)
        self.current_wheel = wheel
        wheel.run(allow_restart=False)  # safer if some other tests failed and didn't clean up
        while wheel.is_running():
            sleep(1)
        assert len(wheel.get_status_report().values()) == 3
        assert all(v == Status.FINISHED for v in wheel.get_status_report().values())
        assert structures.count("{}") == len(paths)
