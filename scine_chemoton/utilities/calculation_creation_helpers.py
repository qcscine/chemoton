#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import scine_database as db
from typing import Optional, List


def finalize_calculation(
    calculation: db.Calculation,
    structure_collection: db.Collection,
    structure_ids: Optional[List[db.ID]] = None
):
    if structure_ids is None:
        structure_ids = calculation.get_structures()
    for s_id in structure_ids:
        structure = db.Structure(s_id, structure_collection)
        structure.add_calculation(calculation.job.order, calculation.id())
    calculation.set_status(db.Status.HOLD)
