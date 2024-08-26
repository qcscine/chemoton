#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import os
import ast
import _ast
import importlib
from collections import UserList
from inspect import isabstract, getmembers
from typing import List, Tuple

import scine_chemoton
from scine_chemoton.gears import Gear
from scine_chemoton.utilities.datastructure_transfer import db_safe_deepcopy, make_picklable


class GearContainer(UserList):
    """
    Automatically gather a list of all non-abstract gears from scine_chemoton.
    """

    def __init__(self) -> None:
        super().__init__()
        class_to_search = Gear

        for path, class_name in self._get_all_classes_names_from_chemoton():
            module = importlib.import_module(path)
            loaded_class = getattr(module, class_name)

            if issubclass(loaded_class, class_to_search) and not isabstract(loaded_class):
                self.data.append(loaded_class)

    def _get_all_classes_names_from_chemoton(self) -> List[Tuple[str, str]]:
        """
        Search for all classes in scine_chemoton module.
        Returns a list of Tuples of module path and class name.
        """
        classes = []
        for path in self._get_files_names_from_chemoton():
            with open(path) as mf:
                tree = ast.parse(mf.read())

            # parse module path
            module_path = path.split("scine_chemoton")[-1]
            module_path = module_path.replace(".py", "")
            module_path = module_path.replace("/__init__", "")
            module_path = "scine_chemoton" + module_path.replace("/", ".")

            # parse classes names
            module_classes = [_ for _ in tree.body if isinstance(_, _ast.ClassDef)]
            classes.extend([(module_path, c.name) for c in module_classes])
        return classes

    @staticmethod
    def _get_files_names_from_chemoton() -> List[str]:
        """
        Return all .py files in scine_chemoton module.
        """
        location = None
        try:
            location = scine_chemoton.__file__  # can also be None
        except AttributeError:
            pass
        if location is None:
            members = getmembers(scine_chemoton)
            for m in members:
                if "__file__" in m and m[1] is not None:
                    location = m[1]
                    break
                if "__path__" in m and m[1] is not None:
                    for possible_location in m[1]:
                        if os.path.exists(possible_location):
                            location = possible_location
                            break
            if location is None:
                here = os.path.abspath(os.path.dirname(__file__))
                location = os.path.join(here, "..", "..", "scine_chemoton", "__init__.py")
                if os.path.exists(location):
                    raise RuntimeError("The location of your installed SCINE Chemoton could not be determined")
        path = os.path.dirname(location)

        py_files = []

        for root, _, files in os.walk(path):
            for f in files:
                if f[-3:] == ".py":
                    py_files.append(os.path.join(root, f))

        return py_files


def test_all_gears_pickleable():
    gears = GearContainer()
    for gear in gears:
        gear_inst = gear()
        assert gear_inst is not None
        assert isinstance(gear_inst, gear)

        gear = make_picklable(gear)
        assert gear_inst is not None
        assert isinstance(gear_inst, gear)

        gear_inst = db_safe_deepcopy(gear_inst)
        assert gear_inst is not None
        assert isinstance(gear_inst, gear)
