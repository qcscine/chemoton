#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from multiprocessing.connection import Connection
from time import sleep
from typing import List, Union

from scine_chemoton.utilities.datastructure_transfer import read_connection, make_picklable, ReadAble
from scine_chemoton.steering_wheel.datastructures import (
    SelectionResult,
    NetworkExpansionResult,
    ExplorationResult,
)


class StopMultipleSplitCommunicationMethod:
    pass


class WaitForFurtherResults:
    pass


def _receive_split_result_from_pipe(pipe: Union[Connection, ReadAble, None]) \
        -> Union[ExplorationResult, StopMultipleSplitCommunicationMethod, None]:
    # we first need to know the class which we want to receive, this is the first sent signal
    cls = read_connection(pipe, return_first_signal=True)
    if isinstance(cls, WaitForFurtherResults):
        # pipe tells us to continuously read until we get a result
        cls = None
        while cls is None:
            sleep(0.1)
            cls = read_connection(pipe, return_first_signal=True)
            if isinstance(cls, WaitForFurtherResults):
                cls = None
            if isinstance(cls, StopMultipleSplitCommunicationMethod):
                # it was just an empty list of results
                return None
    if cls is None or isinstance(cls, StopMultipleSplitCommunicationMethod):
        return cls
    if cls not in [SelectionResult, NetworkExpansionResult]:
        raise ValueError(f"Failed to receive the class information for receiving a split result; "
                         f"received {cls} instead.")
    return cls.receive_from_pipe(pipe)


def receive_multiple_results_from_pipe(pipe: Union[Connection, ReadAble, None]) -> List[ExplorationResult]:
    if pipe is None:
        return []

    def _impl() -> List[ExplorationResult]:
        results = []
        first_iteration = True
        while True:
            result = _receive_split_result_from_pipe(pipe)
            # we only accept None as a break condition for the first iteration for the case that nothing is in the pipe
            # otherwise this is likely a race condition problem and we only stop reading the results
            # based on the appropriate signal because we know that something has been sent already
            # this avoids to cut result lists apart
            if (result is None and first_iteration) or isinstance(result, StopMultipleSplitCommunicationMethod):
                break
            if result is None:
                # not first iteration, but pipe still missing other parts, just wait
                continue
            first_iteration = False
            results.append(result)
        return results

    # in connection, there might be a series of result lists, because the worker always sends a complete list
    # after each step.
    # therefore, we return the last result list
    results_to_return = _impl()
    while True:
        next_results = _impl()
        if len(next_results) == 0:
            # if there are no more results, we get an empty results list
            return results_to_return
        results_to_return = next_results


def send_multiple_results_in_pipe(results: List[ExplorationResult], pipe: Connection) -> None:
    results_to_send = make_picklable(results)
    if results_to_send is not None:
        for result in results_to_send:
            if result is None:
                pipe.send(None)
                continue
            assert isinstance(result, ExplorationResult)
            result.send_in_pipe(pipe)
    pipe.send(StopMultipleSplitCommunicationMethod())
