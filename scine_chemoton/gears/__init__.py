#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from abc import ABC, abstractmethod
from collections import UserDict
from ctypes import c_int
from enum import Enum
from setproctitle import setproctitle
from typing import List, Callable, Optional
import signal
import time

# Third party imports
import scine_database as db

from scine_chemoton.utilities import connect_to_db
from scine_chemoton.utilities.options import BaseOptions


class HoldsCollections:

    def __init__(self):
        super().__init__()  # necessary for multiple inheritance
        self._required_collections: List[str] = []
        self._manager = None
        self._calculations = None
        self._compounds = None
        self._elementary_steps = None
        self._flasks = None
        self._properties = None
        self._reactions = None
        self._structures = None

    @staticmethod
    def possible_attributes() -> List[str]:
        return [
            "manager",
            "calculations",
            "compounds",
            "elementary_steps",
            "flasks",
            "properties",
            "reactions",
            "structures",
        ]

    def initialize_collections(self, manager: db.Manager) -> None:
        for attr in self._required_collections:
            if attr not in self.possible_attributes():
                raise NotImplementedError(f"The initialization of member {attr} for class {self.__class__.__name__}, "
                                          f"is not possible, we are only supporting {self.possible_attributes()}.")
        for attr in self._required_collections:
            if attr == "manager":
                setattr(self, f"_{attr}", manager)
            else:
                setattr(self, f"_{attr}", manager.get_collection(attr))

    def unset_collections(self) -> None:
        if hasattr(self, "_parent"):
            setattr(self, "_parent", None)
        for attr in self.possible_attributes():
            setattr(self, f"_{attr}", None)
        self._unset_collections_of_attributes(self)

    def _unset_collections_of_attributes(self, inst):
        if isinstance(inst, dict) or isinstance(inst, UserDict):
            items = inst.items()
        elif hasattr(inst, '__dict__'):
            items = inst.__dict__.items()
        elif hasattr(inst, '__slots__'):
            items = ((s, getattr(inst, s)) for s in inst.__slots__)
        else:
            return inst
        for key, attr in list(items):
            if isinstance(attr, HoldsCollections):
                attr.unset_collections()
            elif isinstance(attr, db.Collection) or isinstance(attr, db.Manager):
                if isinstance(inst, dict) or isinstance(inst, UserDict):
                    inst[key] = None
                else:
                    setattr(inst, key, None)
                continue
            if isinstance(attr, Enum):
                continue
            if hasattr(attr, '__dict__') or hasattr(attr, "__slots__") \
                    or isinstance(attr, dict) or isinstance(attr, UserDict):
                attr = self._unset_collections_of_attributes(attr)
                if isinstance(attr, dict) or isinstance(attr, UserDict):
                    if isinstance(inst, dict) or isinstance(inst, UserDict):
                        inst[key] = attr
                    else:
                        setattr(inst, key, attr)
                    continue
            if hasattr(attr, '__iter__') and hasattr(attr, "__setitem__") and not isinstance(attr, str):
                for i, a in list(enumerate(attr)):
                    a = self._unset_collections_of_attributes(a)
                    attr[i] = a
            if isinstance(inst, dict) or isinstance(inst, UserDict):
                inst[key] = attr
            else:
                setattr(inst, key, attr)
        return inst


class HasName:

    def __init__(self):
        super().__init__()  # necessary for multiple inheritance
        self._name = 'Chemoton' + self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, n: str):
        self._name = n

    def _give_current_process_own_name(self) -> None:
        setproctitle(self.name)

    def _remove_chemoton_from_name(self) -> None:
        self._name = self._name.replace("Chemoton", "")

    def _join_names(self, objects: list) -> None:
        self._name += "(" + ", ".join(f.name for f in objects) + ")"

    def __str__(self):
        return self.name


class Gear(ABC, HoldsCollections, HasName):
    """
    The base class for all Gears.

    A Gear in Chemoton is a continuous loop that alters, analyzes or interacts
    in any other way with a reaction network stored in a SCINE Database.
    Each Gear has to be attached to an Engine(:class:`scine_chemoton.engine.Engine`)
    and will then help in driving the exploration of chemical reaction networks
    forward.

    Extending the features of Chemoton can be done by adding new Gears or altering
    existing ones.
    """

    class Options(BaseOptions):

        __slots__ = ("model", "cycle_time")

        def __init__(self):
            super().__init__()
            self.model = db.Model("PM6", "PM6", "")
            """
            Model
                The model the Gear is working with.
            """
            self.cycle_time: int = 10
            """
            int
                The minimum number of seconds between two cycles of the Gear.
                Cycles are finished independently of this option, hence if a cycle
                takes longer than the cycle_time will effectively lead to longer
                cycle times and not cause multiple cycles of the same Gear.
            """

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self.name = 'Chemoton' + self.__class__.__name__ + 'Gear'
        self.stop_at_next_break_point = False

    def __eq__(self, other):
        if not isinstance(other, Gear):
            return False
        return self.stop_at_next_break_point == other.stop_at_next_break_point \
            and self.options == other.options

    class _DelayedKeyboardInterrupt:
        def __init__(self, callable: Optional[Callable]):
            self.signal_received = False
            self.callable = callable

        def __enter__(self):
            self.old_handler = \
                signal.signal(signal.SIGINT, self.handler)  # pylint: disable=attribute-defined-outside-init

        def handler(self, sig, frame):
            self.signal_received = (sig, frame)
            if self.callable is not None:
                self.callable()

        def __exit__(self, type, value, traceback):
            signal.signal(signal.SIGINT, self.old_handler)
            if self.signal_received:
                self.old_handler(*self.signal_received)

    def __call__(self, credentials: db.Credentials, loop_count: c_int, single: bool = False):
        """
        Starts the main loop of the Gear, then acting on the database referenced
        by the given credentials.

        Parameters
        ----------
        credentials :: db.Credentials (Scine::Database::Credentials)
            The credentials to a database storing a reaction network.
        loop_count :: c_int
            A shared memory integer that allows to communicate the number of loops
            across processes.
        single :: bool
            If true, runs only a single iteration of the actual loop.
            Default: false, meaning endless repetition of the loop.
        """
        self._give_current_process_own_name()

        # Make sure cycle time exists
        sleep_time = self.options.cycle_time

        # Prepare database connection and members
        _initialize_a_gear_to_a_db(self, credentials)

        # Infinite loop with sleep
        last_cycle = time.time()
        # Instant first loop
        with self._DelayedKeyboardInterrupt(callable=self.stop):
            self._loop_impl()
        loop_count.value += 1
        # Stop if only a single loop was requested
        if single:
            return
        while True:
            # Wait if needed
            now = time.time()
            if now - last_cycle < sleep_time:
                time.sleep(sleep_time - now + last_cycle)
            last_cycle = time.time()

            with self._DelayedKeyboardInterrupt(callable=self.stop):
                self._loop_impl()
            loop_count.value += 1

    @abstractmethod
    def _loop_impl(self):  # Main loop to be implemented by all derived Gears.
        pass

    def _propagate_db_manager(self, manager: db.Manager):
        pass

    def stop(self) -> None:
        self.stop_at_next_break_point = True


def _initialize_a_gear_to_a_db(gear: Gear, credentials: db.Credentials) -> None:
    if gear._manager is None or gear._manager.get_credentials() != credentials:
        gear._manager = connect_to_db(credentials)
        # Get required collections
        gear.initialize_collections(gear._manager)
    # always propagate in case a member has been changed
    gear._propagate_db_manager(gear._manager)
