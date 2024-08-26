#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import List, Optional
from typing_extensions import Self

from wrapt import ObjectProxy
import scine_database as db


def _place_hold_arguments() -> List[str]:
    return ["place", "holder", "model"]


class ModelNotSetError(Exception):
    pass


class PlaceHolderModelType(db.Model):
    """
    The type to check if a model is still a place-holder electronic structure model.
    Do not use to construct a place-holder model!
    """

    def __init__(self, *args, know_what_i_am_doing: bool = False, **kwargs) -> None:
        super().__init__(*_place_hold_arguments())
        if not know_what_i_am_doing:
            raise PermissionError("Do not construct a place-holder model directly, "
                                  "use 'construct_place_holder_model()' instead.")


class _PlaceHolderModelProxy(ObjectProxy):
    """
    Facilitates the class change
    """

    _ref_place_holder = db.Model(*_place_hold_arguments())
    __wrapped__: PlaceHolderModelType

    def __init__(self, *args, pre_existing_model: Optional[db.Model] = None, **kwargs) -> None:
        if pre_existing_model is not None:
            super().__init__(pre_existing_model)
        else:
            super().__init__(PlaceHolderModelType(know_what_i_am_doing=True))

    @classmethod
    def from_existing_model(cls, model: db.Model) -> Self:
        return cls(pre_existing_model=model)

    @property  # type: ignore
    def __class__(self):
        if self == self._ref_place_holder:
            return PlaceHolderModelType
        return db.Model

    def __getstate__(self):
        return self.__wrapped__.__getstate__()

    def __setstate__(self, state):
        self.__wrapped__.__setstate__(state)

    def __copy__(self):
        instance = _PlaceHolderModelProxy.__new__(_PlaceHolderModelProxy)
        instance.__wrapped__ = self.__wrapped__.__copy__()
        return instance

    def __deepcopy__(self, memo):
        instance = _PlaceHolderModelProxy.__new__(_PlaceHolderModelProxy)
        instance.__wrapped__ = self.__wrapped__.__deepcopy__(memo)
        return instance

    def __reduce__(self):
        return _PlaceHolderModelProxy.from_existing_model, (self.__wrapped__,)

    def __reduce_ex__(self, protocol):
        return _PlaceHolderModelProxy.from_existing_model, (self.__wrapped__,)


def construct_place_holder_model():
    """
    Construct a place-holder model

    Examples
    --------
    >>> place_holder = construct_place_holder_model()
    >>> assert isinstance(place_holder, db.Model)
    >>> assert isinstance(place_holder, PlaceHolderModelType)
    >>> # if we now change fields, it is no longer a place-holder
    >>> place_holder.method_family = 'DFT'
    >>> assert isinstance(place_holder, db.Model)
    >>> assert not isinstance(place_holder, PlaceHolderModelType)
    """
    # this method hides the actually used Proxy
    # so that people do not use the proxy in the type check
    # because isinstance(place_holder, _PlaceHolderModelProxy) would always evaluate to True
    return _PlaceHolderModelProxy()
