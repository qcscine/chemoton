#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""


# Standard library imports
import multiprocessing
from typing import Any, Callable, List, Optional
import signal
import sys

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
    credentials : db.Credentials (Scine::Database::Credentials)
        The credentials to a database storing a reaction network.
        The started process will connect and interact with the referenced
        database.
    fork : bool
        If true, this will cause the Engine to start the process defined by
        the given Gear with a fork, meaning it will run in a separate
        thread.
    """

    def __init__(self, credentials: db.Credentials, fork: bool = True) -> None:
        self._credentials = credentials
        self._fork = fork
        self._gear: Optional[Gear] = None
        self._proc: Optional[multiprocessing.Process] = None
        self._loop_count: Optional[Any] = multiprocessing.Value('i', 0)

    def set_gear(self, gear: Gear) -> None:
        """
        Parameters
        ----------
        gear : Gear (scine_chemoton.gears.Gears)
            The gear to be used when starting this engine.
        """
        self._gear = gear

    def run(self, single: bool = False):
        """
        Starts turning the given Gear (:class:`scine_chemoton.gears.Gears`).

        Parameters
        ----------
        single : bool
            If true, runs only a single iteration of the Gear's loop.
            Default: false, meaning endless repetition of the loop.

        Raises
        ------
        AttributeError
            If no gear was added to the engine, prior to starting it.
        """
        if not self._gear:
            raise AttributeError("Engine has not received a gear")
        self._gear.stop_at_break_point(False)
        if self._loop_count is None:
            self._loop_count = multiprocessing.Value('i', 0)
        if self._fork and self._gear is not None:
            self._proc = multiprocessing.Process(
                target=self._gear, args=(self._credentials, self._loop_count), kwargs={"single": single}
            )
            self._proc.start()
        elif self._gear is not None:
            self._gear(self._credentials, self._loop_count, single=single)  # type: ignore

    def is_running(self) -> bool:
        """
        Returns
        -------
        bool
            True if the engine is running, False otherwise.
        """
        if self._proc is None:
            return False
        return self._proc.is_alive()

    def stop(self) -> None:
        """
        In case of a forked job, this will send an interrupt signal to the forked process, leading to a graceful exit.
        """
        if self._gear is not None:
            self._gear.stop()

    def join(self, timeout: Optional[int] = None):
        """
        In case of a forked job this will wait for the graceful exit of the job.
        If the process has not been signalled to stop, this will also initiate the stop.
        """
        if self._fork and self._proc:
            if self._gear is not None and not self._gear.have_to_stop_at_next_break_point():
                self.stop()
        self._cleanup(timeout)

    def terminate(self) -> None:
        """
        In case of a forked job, this will terminate the forked process.

        Notes
        -----
        The job will **NOT** be stopped gracefully.
        """
        if self._fork and self._proc:
            self._proc.terminate()
            self._cleanup(timeout=1)
        else:
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


class EngineHandler:
    """
    A class that takes a list of engines and an optional list of signals.
    It takes care that all given engines are stopped and their forked processes
    are joined when one of the signals is sent to the process.
    No further actions are required, the class must simply be kept in scope.
    """

    def __init__(self, engines: List[Engine], signals: Optional[List[signal.Signals]] = None) -> None:
        if signals is None:
            signals = [signal.SIGINT, signal.SIGTERM]
        self._engines = engines
        for s in signals:
            signal.signal(s, lambda _, __: self._handler())

    def run(self, single: bool = False) -> None:
        for engine in self._engines:
            engine.run(single=single)

    def wait_for_stop_signal(self, input_: Callable[[str], str] = input) -> None:
        answer = ""
        while answer != "stop":
            answer = input_("Enter 'stop' if you want to stop all running engines: ")
        self.stop_and_join()

    def stop_and_join(self) -> None:
        for engine in self._engines:
            engine.stop()
        for engine in self._engines:
            engine.join()

    def _handler(self) -> None:
        self.stop_and_join()
        sys.exit(1)

    def __iadd__(self, other: Engine):
        self._engines.append(other)
        return self

    def append(self, other: Engine):
        self._engines.append(other)

    def __len__(self):
        return len(self._engines)
