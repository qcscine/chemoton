#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from copy import copy
from tempfile import NamedTemporaryFile
import pickle


from scine_chemoton.utilities.place_holder_model import (
    PlaceHolderModelType,
    construct_place_holder_model,
    _place_hold_arguments
)

import scine_database as db
import pytest


def test_forbidden_init():
    with pytest.raises(PermissionError):
        PlaceHolderModelType()


def test_invalid_type_for_base():
    model = db.Model(*_place_hold_arguments())
    assert isinstance(model, db.Model)
    assert not isinstance(model, PlaceHolderModelType)
    model.method_family = 'DFT'
    assert isinstance(model, db.Model)
    assert not isinstance(model, PlaceHolderModelType)


def test_type_alteration():
    place_holder = construct_place_holder_model()
    assert isinstance(place_holder, db.Model)
    assert isinstance(place_holder, PlaceHolderModelType)
    # if we now change fields, it is no longer a place-holder
    place_holder.method_family = 'DFT'
    assert isinstance(place_holder, db.Model)
    assert not isinstance(place_holder, PlaceHolderModelType)


def test_copy():
    place_holder = construct_place_holder_model()
    place_holder_copy = copy(place_holder)
    assert place_holder == place_holder_copy
    assert place_holder is not place_holder_copy
    assert isinstance(place_holder_copy, db.Model)
    assert isinstance(place_holder_copy, PlaceHolderModelType)

    # if we now change fields, it is no longer a place-holder
    place_holder.method_family = 'DFT'
    assert isinstance(place_holder, db.Model)
    assert not isinstance(place_holder, PlaceHolderModelType)
    place_holder_copy = copy(place_holder)
    assert place_holder == place_holder_copy
    assert place_holder is not place_holder_copy
    assert isinstance(place_holder_copy, db.Model)
    assert not isinstance(place_holder_copy, PlaceHolderModelType)


def test_pickle():
    place_holder = construct_place_holder_model()
    with NamedTemporaryFile("w+b", suffix=".pkl") as f:
        pickle.dump(place_holder, f)
        f.seek(0)
        read_place_holder = pickle.load(f)
    assert place_holder == read_place_holder
    assert place_holder is not read_place_holder
    assert isinstance(read_place_holder, db.Model)
    assert isinstance(read_place_holder, PlaceHolderModelType)
    place_holder.method_family = 'DFT'
    assert isinstance(place_holder, db.Model)
    assert not isinstance(place_holder, PlaceHolderModelType)
    read_place_holder.method_family = 'DFT'
    assert isinstance(read_place_holder, db.Model)
    assert not isinstance(read_place_holder, PlaceHolderModelType)
    with NamedTemporaryFile("w+b", suffix=".pkl") as f:
        pickle.dump(place_holder, f)
        f.seek(0)
        read_place_holder = pickle.load(f)
    assert place_holder == read_place_holder
    assert place_holder is not read_place_holder
    assert isinstance(read_place_holder, db.Model)
    assert not isinstance(read_place_holder, PlaceHolderModelType)
