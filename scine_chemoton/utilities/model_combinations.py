#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Optional

import scine_database as db


class ModelCombination:
    """
    This class combines two electronic structure models to a model combination. One of the models is supposed to be
    used to calculate structures, hessians, and gradients, while the second model is used to calculate the electronic
    energy.

    Parameters
    ----------
    electronic_model : db.Model
        The model for the electronic energies.
    hessian_model : Optional[db.Model]
        The model for the structures, hessians, and gradients. If None is provided, the electronic_model is used.
    """

    def __init__(self, electronic_model: db.Model, hessian_model: Optional[db.Model] = None) -> None:
        self.electronic_model: db.Model = electronic_model
        if hessian_model is None:
            hessian_model = electronic_model
        self.hessian_model: db.Model = hessian_model
        assert self.hessian_model
        assert isinstance(self.electronic_model, db.Model)
        assert isinstance(self.hessian_model, db.Model)

    def __eq__(self, other):
        """
        Equal operator. Electronic energy and Hessian model must be the identical.
        """
        if not isinstance(other, ModelCombination):
            return False
        return self.electronic_model == other.electronic_model and self.hessian_model == other.hessian_model
