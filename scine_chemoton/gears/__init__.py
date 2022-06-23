#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import signal
import time

# Third party imports
import scine_database as db


class Gear:
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
        self._calculations = None
        self._compounds = None
        self._reactions = None
        self._elementary_steps = None
        self._structures = None
        self._properties = None
        self.name = 'Chemoton' + self.__class__.__name__ + 'Gear'

    class _DelayedKeyboardInterrupt:
        def __enter__(self):
            self.signal_received = False
            self.old_handler = signal.signal(signal.SIGINT, self.handler)

        def handler(self, sig, frame):
            self.signal_received = (sig, frame)

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

        # Prepare clean database
        self.manager = db.Manager()
        self.manager.set_credentials(credentials)
        self.manager.connect()
        time.sleep(1.0)
        if not self.manager.has_collection("calculations"):
            raise RuntimeError("Stopping Gear/Engine: database is missing collections.")

        # Get required collections
        if hasattr(self, "_calculations"):
            if self._calculations:
                self._calculations = self.manager.get_collection("calculations")
        if hasattr(self, "_compounds"):
            if self._compounds:
                self._compounds = self.manager.get_collection("compounds")
        if hasattr(self, "_reactions"):
            if self._reactions:
                self._reactions = self.manager.get_collection("reactions")
        if hasattr(self, "_elementary_steps"):
            if self._elementary_steps:
                self._elementary_steps = self.manager.get_collection("elementary_steps")
        if hasattr(self, "_structures"):
            if self._structures:
                self._structures = self.manager.get_collection("structures")
        if hasattr(self, "_properties"):
            if self._properties:
                self._properties = self.manager.get_collection("properties")

        self._propagate_db_manager(self.manager)

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
