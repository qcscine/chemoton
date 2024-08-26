#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
# Standard library imports
from abc import ABCMeta, abstractmethod
from functools import wraps
from typing import Callable, Dict, List, Tuple, Optional, Any
from warnings import warn

# Third party imports
from numpy import ndarray
import scine_database as db
from scine_utilities import ValueCollection

from scine_chemoton.gears import HoldsCollections
from scine_chemoton.gears import Gear
from scine_chemoton.filters.reactive_site_filters import ReactiveSiteFilter
from scine_chemoton.utilities.options import BaseOptions
from scine_chemoton.utilities.place_holder_model import (
    construct_place_holder_model,
    PlaceHolderModelType
)
from scine_chemoton.utilities.warnings import ModelChangedWarning


class TrialGenerator(HoldsCollections, metaclass=ABCMeta):
    """
    Base class for elementary step trial generators
    """

    class Options(BaseOptions):
        """
        The options of the TrialGenerator
        """
        __slots__ = ("_parent", "model", "base_job_settings")

        def __init__(self, parent: Optional[Any] = None) -> None:
            self._parent = parent  # best be first member to be set, due to __setattr__
            super().__init__()
            self.model: db.Model = construct_place_holder_model()
            self.base_job_settings: ValueCollection = ValueCollection({})

        def __setattr__(self, item, value):
            """
            Overwritten standard method to synchronize model option
            """
            model_case = bool(
                item == "model" and
                hasattr(self, "model") and
                isinstance(value, db.Model) and
                not isinstance(value, PlaceHolderModelType) and
                self.model != value and
                hasattr(self, "_parent") and
                self._parent is not None and
                hasattr(self._parent, "_parent") and
                hasattr(self._parent._parent, "options") and
                hasattr(self._parent._parent.options, "model") and
                self._parent._parent.options.model != value
            )
            super().__setattr__(item, value)
            if model_case:
                if not isinstance(self._parent._parent.options.model, PlaceHolderModelType):
                    warn("The model of the ElementaryStepGear is overwritten by the TrialGenerator.",
                         category=ModelChangedWarning)
                self._parent._parent.options.model = value

    def __init__(self) -> None:
        super().__init__()
        self._parent: Optional[Gear] = None  # allows to propagate model information to gear
        self.options = self.Options(parent=self)
        self._required_collections = ["calculations", "structures"]
        self.reactive_site_filter = ReactiveSiteFilter()

    @abstractmethod
    def clear_cache(self):
        raise NotImplementedError

    @abstractmethod
    def unimolecular_reactions(self, structure: db.Structure, with_exact_settings_check: bool = False) -> None:
        """
        Creates reactive complex calculations corresponding to the unimolecular
        reactions of the structure if there is not already a calculation to
        search for a reaction of the same structure with the same job order.

        Parameters
        ----------
        structure : db.Structure
            The structure to be considered. The Structure has to
            be linked to a database.
        with_exact_settings_check : bool
            If True, more expensive queries are carried out to check if the settings of the
            calculations are exactly the same as the settings of the trial generator. This allows to add more
            inclusive additional reaction trials but the queries are less efficient, therefore this option
            should be only toggled if necessary.
        """
        raise NotImplementedError

    @abstractmethod
    def bimolecular_reactions(self, structure_list: List[db.Structure], with_exact_settings_check: bool = False) \
            -> None:
        """
        Creates reactive complex calculations corresponding to the bimolecular
        reactions between the structures if there is not already a calculation
        to search for a reaction of the same structures with the same job order.

        Parameters
        ----------
        structure_list : List[db.Structure]
            List of the two structures to be considered.
            The Structures have to be linked to a database.
        with_exact_settings_check : bool
            If True, more expensive queries are carried out to check if the settings of the
            calculations are exactly the same as the settings of the trial generator. This allows to add more
            inclusive additional reaction trials but the queries are less efficient, therefore this option
            should be only toggled if necessary.
        """
        raise NotImplementedError

    @abstractmethod
    def unimolecular_coordinates(self, structure: db.Structure, with_exact_settings_check: bool = False) \
            -> List[Tuple[List[List[Tuple[int, int]]], int]]:
        """
        Returns the reaction coordinates allowed for unimolecular reactions for the given structure based on
        the set options and filters. This method does not set up new calculations.
        The returned object is a list of tuple.
        The first argument in the tuple is a list of reaction coordinates.
        The second argument in the tuple is the number of dissociations.

        Parameters
        ----------
        structure : db.Structure
            The structure to be considered. The Structure has to
            be linked to a database.
        with_exact_settings_check : bool
            If True, more expensive queries are carried out to check if the settings of the
            calculations are exactly the same as the settings of the trial generator. This allows to add more
            inclusive additional reaction trials but the queries are less efficient, therefore this option
            should be only toggled if necessary.
        """
        raise NotImplementedError

    @abstractmethod
    def bimolecular_coordinates(self, structure_list: List[db.Structure], with_exact_settings_check: bool = False) \
            -> Dict[
        Tuple[List[Tuple[int, int]], int],
        List[Tuple[ndarray, ndarray, float, float]]
    ]:
        """
        Returns the reaction coordinates allowed for bimolecular reactions for the given structures based on
        the set options and filters. This method does not set up new calculations.
        The returned object is a list of dictionary.
        The keys are a tuple containing a reaction coordinates and the number of dissociations.
        The values hold a list of instructions. Each entry in this list allows to construct a reactive complex.
        Therefore, the number of reactive complexes per reaction coordinate can also be inferred.

        Notes
        -----
        The index basis (total system or separate systems) of the returned indices in the reaction coordinates
        varies between different implementations!

        Parameters
        ----------
        structure_list : List[db.Structure]
            List of the two structures to be considered.
            The Structures have to be linked to a database.
        with_exact_settings_check : bool
            If True, more expensive queries are carried out to check if the settings of the
            calculations are exactly the same as the settings of the trial generator. This allows to add more
            inclusive additional reaction trials but the queries are less efficient, therefore this option
            should be only toggled if necessary.
        """
        raise NotImplementedError

    def _sanity_check_configuration(self):
        if not isinstance(self.reactive_site_filter, ReactiveSiteFilter):
            raise TypeError(f"Expected a ReactiveSiteFilter (or a class derived from it) "
                            f"in {self.__class__.__name__}.reactive_site_filter.")

    @abstractmethod
    def get_unimolecular_job_order(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_bimolecular_job_order(self) -> str:
        raise NotImplementedError

    def _get_settings(self, settings: ValueCollection) -> ValueCollection:
        """
        Convenience method to combine given settings with the base settings.
        The given settings have precedence over the base_job_settings of the instance.
        """
        return ValueCollection({**self.options.base_job_settings, **settings})  # type: ignore


def _sanity_check_wrapper(fun: Callable) -> Callable:

    @wraps(fun)
    def _impl(self: TrialGenerator, *args, **kwargs):
        self._sanity_check_configuration()
        result = fun(self, *args, **kwargs)
        return result

    return _impl
