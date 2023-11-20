#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import unittest

# Third party imports
import scine_utilities as utils
from scine_database import test_database_setup as db_setup

# Local application imports
from scine_chemoton.gears.compound import BasicAggregateHousekeeping
from scine_chemoton.gears.thermo import BasicThermoDataCompletion
from scine_chemoton.steering_wheel.datastructures import Status, GearOptions, ProtocolEntry
from scine_chemoton.steering_wheel.network_expansions import GiveWholeDatabaseWithModelResult
from scine_chemoton.steering_wheel.network_expansions.basics import SimpleOptimization, ThermochemistryGeneration


class BasicEnhancementsTests(unittest.TestCase):

    def setUp(self) -> None:
        self.manager = db_setup.get_clean_db(f"chemoton_{self.__class__.__name__}")
        self.model = db_setup.get_fake_model()
        self.credentials = self.manager.get_credentials()

    def tearDown(self) -> None:
        self.manager.wipe()

    def test_standard_inits(self):
        for cls in [GiveWholeDatabaseWithModelResult, SimpleOptimization, ThermochemistryGeneration]:
            inst = cls(self.model)
            assert inst.options.model == self.model
            assert inst.status == Status.WAITING
            assert not inst.protocol
            inst.dry_setup_protocol(self.credentials)
            assert inst.protocol

    def test_gear_options(self):
        dummy_opt = SimpleOptimization(self.model)
        dummy_opt.dry_setup_protocol(self.credentials)
        assert dummy_opt.protocol
        gears = [p.gear for p in dummy_opt.protocol if isinstance(p, ProtocolEntry)]
        options = GearOptions(gears, self.model)
        options["ChemotonBasicAggregateHousekeepingGear"][0].bond_order_settings = \
            utils.ValueCollection({'only_distance_connectivity': True})
        opt = SimpleOptimization(self.model, include_thermochemistry=True, gear_options=options)
        opt.dry_setup_protocol(self.credentials)
        assert opt.protocol
        assert any(isinstance(p.gear, BasicAggregateHousekeeping) for p in opt.protocol if isinstance(p, ProtocolEntry))
        for p in opt.protocol:
            if isinstance(p, ProtocolEntry) and isinstance(p.gear, BasicAggregateHousekeeping):
                assert p.gear.options.bond_order_settings
                assert p.gear.options.bond_order_settings['only_distance_connectivity']
        assert any(isinstance(p.gear, BasicThermoDataCompletion) for p in opt.protocol if isinstance(p, ProtocolEntry))
