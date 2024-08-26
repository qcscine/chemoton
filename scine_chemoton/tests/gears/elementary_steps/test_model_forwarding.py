#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import pytest

from scine_chemoton.gears.elementary_steps.minimal import MinimalElementarySteps
from scine_chemoton.gears.elementary_steps.trial_generator.bond_based import BondBased
import scine_database as db


def test_model_forwarding_only_set_tg():
    model = db.Model("desired", "model", "")
    elementary_step_gear = MinimalElementarySteps()
    elementary_step_gear.trial_generator = BondBased()
    with pytest.warns(UserWarning):
        elementary_step_gear.trial_generator.options.model = model
    assert elementary_step_gear.options.model == model
    assert elementary_step_gear.trial_generator.options.model == model
    assert elementary_step_gear.options.model == elementary_step_gear.trial_generator.options.model


def test_model_forwarding_only_tg():
    model = db.Model("desired", "model", "")
    elementary_step_gear = MinimalElementarySteps()
    tg = BondBased()
    tg.options.model = model
    with pytest.warns(UserWarning):
        elementary_step_gear.trial_generator = tg
    assert elementary_step_gear.options.model == model
    assert elementary_step_gear.trial_generator.options.model == model
    assert elementary_step_gear.options.model == elementary_step_gear.trial_generator.options.model


def test_model_forwarding_only_set_tg_separate_construction():
    model = db.Model("desired", "model", "")
    elementary_step_gear = MinimalElementarySteps()
    tg = BondBased()
    elementary_step_gear.trial_generator = tg
    with pytest.warns(UserWarning):
        elementary_step_gear.trial_generator.options.model = model
    assert elementary_step_gear.options.model == model
    assert elementary_step_gear.trial_generator.options.model == model
    assert elementary_step_gear.options.model == elementary_step_gear.trial_generator.options.model


def test_model_forwarding_gear_before_updating_tg():
    model = db.Model("desired", "model", "")
    elementary_step_gear = MinimalElementarySteps()
    elementary_step_gear.options.model = model
    tg = BondBased()
    elementary_step_gear.trial_generator = tg
    assert elementary_step_gear.options.model == model
    assert elementary_step_gear.trial_generator.options.model == model
    assert elementary_step_gear.options.model == elementary_step_gear.trial_generator.options.model


def test_model_forwarding_gear_same_model_on_both():
    model = db.Model("desired", "model", "")
    elementary_step_gear = MinimalElementarySteps()
    tg = BondBased()
    elementary_step_gear.trial_generator = tg
    with pytest.warns(UserWarning):
        elementary_step_gear.trial_generator.options.model = model
    elementary_step_gear.options.model = model


def test_model_forwarding_gear_different_model_on_both():
    model = db.Model("desired", "model", "")
    elementary_step_gear = MinimalElementarySteps()
    tg = BondBased()
    elementary_step_gear.trial_generator = tg
    with pytest.warns(UserWarning):
        elementary_step_gear.trial_generator.options.model = model
    assert elementary_step_gear.options.model == model
    assert elementary_step_gear.trial_generator.options.model == model
    assert elementary_step_gear.options.model == elementary_step_gear.trial_generator.options.model

    else_model = db.Model("something", "else", "")
    with pytest.warns(UserWarning):
        elementary_step_gear.options.model = else_model
    assert elementary_step_gear.options.model == else_model
    assert elementary_step_gear.trial_generator.options.model == else_model
    assert elementary_step_gear.options.model == elementary_step_gear.trial_generator.options.model
