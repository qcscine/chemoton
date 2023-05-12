#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""


# Standard library imports
import multiprocessing
from typing import Any, Optional
import signal
import os

# Third party imports
import scine_database as db

# Local application imports
from .gears import Gear


class Engine:
    """
    The Engine is a small class starting and stopping potentially infinitely
    running code acting on a reaction network in a database.
    All continuous jobs are called Gears (:class:`scine_chemoton.gears.Gears`), see
    the appropriate part of this documentation for existing examples.

    Parameters
    ----------
    credentials :: db.Credentials (Scine::Database::Credentials)
        The credentials to a database storing a reaction network.
        The started process will connect and interact with the referenced
        database.
    fork :: bool
        If true, this will cause the Engine to start the process defined by
        the given Gear with a fork, meaning it will run in a separate
        thread.
    """

    def __init__(self, credentials: db.Credentials, fork: bool = True):
        self._credentials = credentials
        self._fork = fork
        self._gear: Optional[Gear] = None
        self._proc: Optional[multiprocessing.Process] = None
        self._loop_count: Optional[Any] = multiprocessing.Value('i', 0)

    def set_gear(self, gear: Gear):
        """
        Parameters
        ----------
        gear :: Gear (scine_chemoton.gears.Gears)
            The gear to be used when starting this engine.
        """
        self._gear = gear

    def run(self, single: bool = False):
        """
        Starts turning the given Gear (:class:`scine_chemoton.gears.Gears`).

        Parameters
        ----------
        single :: bool
            If true, runs only a single iteration of the Gear's loop.
            Default: false, meaning endless repetition of the loop.

        Raises
        ------
        AttributeError
            If no gear was added to the engine, prior to starting it.
        """
        if not self._gear:
            raise AttributeError
        if self._loop_count is None:
            self._loop_count = multiprocessing.Value('i', 0)
        if self._fork and self._gear is not None:
            self._proc = multiprocessing.Process(
                target=self._gear, args=(self._credentials, self._loop_count), kwargs={"single": single}
            )
            self._proc.start()
        elif self._gear is not None:
            self._gear(self._credentials, self._loop_count, single=single)  # type: ignore

    def stop(self):
        """
        In case of a forked job this will send an interrupt signal to the forked process, leading to a graceful exit.
        """
        if self._fork and self._proc:
            self._gear.stop()
            pid = self._proc.pid
            if pid is not None:
                os.kill(pid, signal.SIGINT)

    def join(self, timeout: Optional[int] = None):
        """
        In case of a forked job this will wait for the graceful exit of the job.
        If the process has not been signalled to stop, this will also initiate the stop.
        """
        if self._fork and self._proc:
            if self._gear is not None and not self._gear.stop_at_next_break_point:
                self.stop()
        self._cleanup(timeout)

    def terminate(self):
        """
        In case of a forked job this will terminate the forked process.

        Notes
        -----
        The job will **NOT** be stopped gracefully.
        """
        if self._fork and self._proc:
            self._proc.terminate()
        self._cleanup()

    def _cleanup(self, timeout: Optional[int] = None):
        if self._proc is not None:
            self._proc.join(timeout)
            self._proc = None
        self._loop_count = None

    def get_number_of_gear_loops(self) -> int:
        if self._gear is None:
            raise AttributeError("Engine has not received a gear")
        if self._loop_count is None:
            return 0
        return self._loop_count.value  # type: ignore
