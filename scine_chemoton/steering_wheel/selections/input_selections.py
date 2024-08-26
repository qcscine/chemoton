#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import abstractmethod
from copy import deepcopy
from os import path
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
from typing_extensions import TypeAlias

import scine_database as db
import scine_utilities as utils

from . import SafeFirstSelection
from ..datastructures import SelectionResult, StructureInformation, LogicCoupling
from scine_chemoton.filters.aggregate_filters import AggregateFilter
from scine_chemoton.filters.reactive_site_filters import ReactiveSiteFilter


class InputSelection(SafeFirstSelection):
    """
    Abstract selection for any selection that adds new structures to the database
    """
    valid_input_type: TypeAlias = Any
    list_valid_input_type: TypeAlias = List[valid_input_type]

    def __init__(self, model: db.Model,  # pylint: disable=keyword-arg-before-vararg
                 structure_information_input: list_valid_input_type,
                 label_of_structure: str = "user_guess",
                 additional_aggregate_filters: Optional[List[AggregateFilter]] = None,
                 additional_reactive_site_filters: Optional[List[ReactiveSiteFilter]] = None,
                 logic_coupling: Union[str, LogicCoupling] = LogicCoupling.AND,
                 *args, **kwargs
                 ) -> None:
        """
        Base class of all Selections that enter new structures into the database

        Parameters
        ----------
        model: db.Model
            The model
        structure_information_input: list_or_single_valid_input_type
            Specify a single valid input (dependent on implemented subclass, or a list of these inputs
        additional_aggregate_filters: Optional[List[AggregateFilter]]
        additional_reactive_site_filters:  Optional[List[ReactiveSiteFilter]]
        """
        super().__init__(model, additional_aggregate_filters, additional_reactive_site_filters, logic_coupling,
                         *args, **kwargs)
        self._structure_information = self._read_in_structures(structure_information_input)
        self._label = self._get_label(label_of_structure)

    def _select(self) -> SelectionResult:
        relevant_ids = []
        for info in self._structure_information:
            if not isinstance(info, StructureInformation):
                raise TypeError(f"DevNote: {self.__class__.__name__} built incorrect structure information.")
            if isinstance(info.geometry, utils.AtomCollection):
                structure = db.Structure(db.ID(), self._structures)
                structure.create(info.geometry, info.charge, info.multiplicity)
                structure.set_model(self.options.model)
            elif isinstance(info.geometry, utils.PeriodicSystem):
                structure = db.Structure(db.ID(), self._structures)
                structure_model = deepcopy(self.options.model)
                structure_model.periodic_boundaries = str(info.geometry.pbc)
                structure.create(info.geometry.atoms, info.charge, info.multiplicity)
                structure.set_model(structure_model)
            else:
                raise TypeError(f"DevNote: {self.__class__.__name__} built incorrect structure geometry information.")
            structure.set_label(self._label)
            relevant_ids.append(structure.id())
        return SelectionResult(structures=relevant_ids)

    @staticmethod
    def _get_label(name: str) -> db.Label:
        for label_name, label in db.Label.__members__.items():
            if label_name.lower() == name.lower():
                return label
        raise ValueError(f"Could not find label '{name.lower()}' in "
                         f"{[lab.lower() for lab in db.Label.__members__.keys()]}.")

    @abstractmethod
    def _read_in_structures(self, structure_information_input: list_valid_input_type) \
            -> List[StructureInformation]:
        pass

    def _sanitize_input(self, structure_information_input: list_valid_input_type) -> List[valid_input_type]:
        try:
            inp = deepcopy(structure_information_input)
        except TypeError as e:  # calculator is not pickleable
            if (not isinstance(structure_information_input, list)
                    and isinstance(structure_information_input, utils.core.Calculator)):
                inp = structure_information_input.clone()
            elif isinstance(structure_information_input, list):
                inp = []
                for i in structure_information_input:
                    if isinstance(i, utils.core.Calculator):
                        inp.append(i.clone())
                    else:
                        inp.append(i)
            else:
                raise e
        if self._sane_single_input(inp):
            inp = [inp]
        elif inp is None or not isinstance(inp, list) or not inp or not all(self._sane_single_input(i) for i in inp):
            raise TypeError(f"Received incorrect structure input '{structure_information_input}', "
                            f"'{self.__class__.__name__}' only supports "
                            f"'{self._type_to_string(self.list_valid_input_type)}'.\n"
                            f"See '_sane_single_input implementation for more details.")
        return inp

    @staticmethod
    def _is_int(i: Union[str, int]):
        return isinstance(i, int) or (isinstance(i, str) and i.isdigit())

    @staticmethod
    def _type_to_string(type_annotation: Any) -> str:
        return str(type_annotation).replace("typing.", "")

    @staticmethod
    @abstractmethod
    def _sane_single_input(inp: valid_input_type) -> bool:
        pass


class FileInputSelection(InputSelection):
    """
    Adds new structures with files
    """
    valid_input_type: TypeAlias = Tuple[str, int, int, str]
    list_valid_input_type: TypeAlias = List[valid_input_type]

    def __init__(self, model: db.Model,  # pylint: disable=keyword-arg-before-vararg
                 structure_information_input: list_valid_input_type,
                 label_of_structure: str = "user_guess",
                 additional_aggregate_filters: Optional[List[AggregateFilter]] = None,
                 additional_reactive_site_filters: Optional[List[ReactiveSiteFilter]] = None,
                 logic_coupling: Union[str, LogicCoupling] = LogicCoupling.AND,
                 *args, **kwargs
                 ) -> None:
        super().__init__(model, structure_information_input, label_of_structure,
                         additional_aggregate_filters, additional_reactive_site_filters, logic_coupling,
                         *args, **kwargs)

    @staticmethod
    def _sane_single_input(inp: valid_input_type) -> bool:
        if isinstance(inp, str) or not hasattr(inp, "__iter__") or len(inp) not in [3, 4]:
            return False
        return isinstance(inp[0], str) and all(InputSelection._is_int(i) for i in inp[1:3]) \
            and (len(inp) == 3 or isinstance(inp[3], str))  # type: ignore

    def _read_in_structures(self, structure_information_input: list_valid_input_type) \
            -> List[StructureInformation]:
        infos = []
        inp = self._sanitize_input(structure_information_input)
        for i in inp:
            path_input = Path(i[0]).expanduser()
            if not path.exists(path_input):
                raise FileNotFoundError(f"The file {path_input} does not exists.")
            atoms = utils.io.read(str(path_input))[0]
            if len(i) == 4 and i[3] and i[3].lower() != "none":
                pbc = utils.PeriodicBoundaries(i[3])
                structure = utils.PeriodicSystem(pbc, atoms)
                infos.append(StructureInformation(geometry=structure, charge=int(i[1]), multiplicity=int(i[2])))
            else:
                infos.append(StructureInformation(geometry=atoms, charge=int(i[1]), multiplicity=int(i[2])))
        return infos


class ScineGeometryInputSelection(InputSelection):
    """
    Add new structures with a Scine Calculator
    """
    valid_input_type: TypeAlias = Union[utils.core.Calculator,
                                        Tuple[Union[utils.AtomCollection, utils.PeriodicSystem], int, int]]
    list_valid_input_type: TypeAlias = List[valid_input_type]

    @staticmethod
    def _sane_single_input(inp: valid_input_type) -> bool:
        if isinstance(inp, utils.core.Calculator):
            return True
        if isinstance(inp, str) or not hasattr(inp, "__iter__") or len(inp) != 3:
            return False
        return (isinstance(inp[0], utils.AtomCollection) or isinstance(inp[0], utils.PeriodicSystem)) and \
            all(InputSelection._is_int(i) for i in inp[1:])

    def _read_in_structures(self, structure_information_input: list_valid_input_type) \
            -> List[StructureInformation]:
        infos = []
        inp = self._sanitize_input(structure_information_input)
        for i in inp:
            if isinstance(i, utils.core.Calculator):
                charge = int(i.settings.get(utils.settings_names.molecular_charge, 0))  # type: ignore
                multiplicity = int(i.settings.get(utils.settings_names.spin_multiplicity, 1))  # type: ignore
                pbc_string = str(i.settings.get(utils.settings_names.periodic_boundaries, ""))  # type: ignore
                if not pbc_string or pbc_string.lower() == "none":
                    geometry: Union[utils.AtomCollection, utils.PeriodicSystem] = i.structure
                else:
                    pbc = utils.PeriodicBoundaries(pbc_string)
                    geometry = utils.PeriodicSystem(pbc, i.structure)
                infos.append(StructureInformation(geometry=geometry, charge=charge, multiplicity=multiplicity))
            else:
                infos.append(StructureInformation(*i))
        return infos
