#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import List
import signal
import time

# Third party imports
import scine_database as db


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


class Gear(HoldsCollections):
    """
    The base class for all Gears.

    A Gear in Chemoton is a continuous loop that alters, analyzes or interacts
    in any other way with a reaction network stored in a SCINE Database.
    Each Gear has to be attached to an Engine(:class:`scine_chemoton.engine.Engine`)
    and will then help in driving the exploration of chemical reaction networks
    forward.

    Extending the features of Chemoton can be done by add new Gears or altering
    existing ones.
    """

    def __init__(self):
        super().__init__()
        self.name = 'Chemoton' + self.__class__.__name__ + 'Gear'

    class _DelayedKeyboardInterrupt:
        def __enter__(self):
            self.signal_received = False  # pylint: disable=attribute-defined-outside-init
            self.old_handler = \
                signal.signal(signal.SIGINT, self.handler)  # pylint: disable=attribute-defined-outside-init

        def handler(self, sig, frame):
            self.signal_received = (sig, frame)  # pylint: disable=attribute-defined-outside-init

        def __exit__(self, type, value, traceback):
            signal.signal(signal.SIGINT, self.old_handler)
            if self.signal_received:
                self.old_handler(*self.signal_received)

    def __call__(self, credentials: db.Credentials, single: bool = False):
        """
        Starts the main loop of the Gear, then acting on the database referenced
        by the given credentials.

        Parameters
        ----------
        credentials :: db.Credentials (Scine::Database::Credentials)
            The credentials to a database storing a reaction network.
        single :: bool
            If true, runs only a single iteration of the actual loop.
            Default: false, meaning endless repetition of the loop.
        """
        try:
            import setproctitle
            setproctitle.setproctitle(self.name)
        except ModuleNotFoundError:
            pass
        # Make sure cycle time exists
        sleep = getattr(getattr(self, "options"), "cycle_time")

        # Prepare database connection
        if self._manager is None or self._manager.get_credentials() != credentials:
            self._manager = db.Manager()
            self._manager.set_credentials(credentials)
            self._manager.connect()
            time.sleep(1.0)
            if not self._manager.has_collection("calculations"):
                raise RuntimeError("Stopping Gear/Engine: database is missing collections.")

            # Get required collections
            self.initialize_collections(self._manager)
            self._propagate_db_manager(self._manager)

        # Infinite loop with sleep
        last_cycle = time.time()
        # Instant first loop
        with self._DelayedKeyboardInterrupt():
            self._loop_impl()
        # Stop if only a single loop was requested
        if single:
            return
        while True:
            # Wait if needed
            now = time.time()
            if now - last_cycle < sleep:
                time.sleep(sleep - now + last_cycle)
            last_cycle = time.time()

            with self._DelayedKeyboardInterrupt():
                self._loop_impl()

    def _loop_impl(self):
        """
        Main loop to be implemented by all derived Gears.
        """
        raise NotImplementedError

    def _propagate_db_manager(self, manager: db.Manager):
        pass
