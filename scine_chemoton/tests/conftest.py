#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""


import pytest


@pytest.fixture(scope='session', autouse=True)
def precondition():
    from scine_database.test_database_setup import get_clean_db
    try:
        _ = get_clean_db()
    except RuntimeError as e:
        pytest.exit(f'{str(e)}\nFirst start database before running unittests.')
