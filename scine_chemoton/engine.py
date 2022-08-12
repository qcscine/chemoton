#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""


# Standard library imports
import multiprocessing
from typing import Optional

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
        if self._fork and self._gear is not None:
            self._proc = multiprocessing.Process(
                target=self._gear, args=(self._credentials,), kwargs={"single": single}
            )
            self._proc.start()
        elif self._gear is not None:
            self._gear(self._credentials, single=single)

    def stop(self):
        """
        In case of a forked job this will terminate the forked process.

        Notes
        -----
        The job will **NOT** be stopped gracefully.
        """
        if self._fork and self._proc:
            self._proc.terminate()
