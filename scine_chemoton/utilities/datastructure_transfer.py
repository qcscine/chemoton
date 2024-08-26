#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import ABC, abstractmethod
from collections import UserDict
from copy import deepcopy
from enum import Enum
from io import UnsupportedOperation, TextIOWrapper, TextIOBase
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from multiprocessing.sharedctypes import SynchronizedBase
from multiprocessing.process import AuthenticationString  # type: ignore
from threading import Thread, Event
import warnings
from time import sleep
from typing import Any, Optional, Union, ItemsView, Tuple, List, Type
from typing_extensions import TypeVar

from scine_database import Collection, Manager
from scine_utilities import AtomCollection

from scine_chemoton.utilities.place_holder_model import PlaceHolderModelType
from scine_chemoton.utilities.warnings import ModelChangedWarning, SettingsChangedWarning


WantedType = TypeVar("WantedType")


def db_safe_deepcopy(obj: WantedType) -> WantedType:
    """
    A deepcopy function which is safe to use with any object that may contain database objects.

    Parameters
    ----------
    obj : Any
        An object to be deepcopied.

    Returns
    -------
    Any
        A deepcopy of the object.
    """
    copied_obj = make_picklable(obj)
    if copied_obj is None:
        raise RuntimeError(f"Deepcopy was not possible {obj}")
    return deepcopy(copied_obj)


def make_picklable(cls: WantedType) -> Optional[WantedType]:
    """
    A function to make an object picklable, by removing all database objects from it.

    Parameters
    ----------
    cls : Any
        An object to be made picklable.

    Notes
    -----
    This function is recursive, and will also make all objects contained in the object picklable.
    However, it does not guarantee that the object is actually picklable, as it does not check for that.

    Returns
    -------
    Optional[Any]
        Returns the object with all database objects removed, or None if the object is not picklable.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=ModelChangedWarning)
        warnings.simplefilter(action="ignore", category=SettingsChangedWarning)
        black_list = [Collection, Manager, SynchronizedBase]
        # add things to white list that have special dunder methods but are completely safe to pickle, see more below
        white_list = [AtomCollection, AuthenticationString, TextIOWrapper, TextIOBase, PlaceHolderModelType]
        if isinstance(cls, Enum) or (hasattr(cls, "__args__") and hasattr(cls, "__origin__")) or isinstance(cls, type):
            # enums and types are safe and wreak havoc on things below.
            return cls  # type: ignore
        if any(isinstance(cls, forbidden) for forbidden in black_list):
            # simply turn to None
            return None
        if any(isinstance(cls, allowed) for allowed in white_list):
            # always safe
            return cls  # type: ignore

        def item_loop(items: ItemsView, inst: Any) -> Any:
            # a loop handling key value pairs
            for k, v in list(items):  # list so we can modify the dict / class on the go
                if isinstance(v, Enum) or (hasattr(v, "__args__") and hasattr(v, "__origin__")) \
                        or isinstance(v, type):
                    # see above
                    continue
                if hasattr(v, "unset_collections"):
                    v.unset_collections()
                v = make_picklable(v)
                # feed back fixed entry
                if isinstance(inst, dict) or isinstance(inst, UserDict):
                    inst[k] = v
                else:
                    setattr(inst, k, v)
            return inst

        if isinstance(cls, dict) or isinstance(cls, UserDict):
            cls = item_loop(cls.items(), cls)
            return cls  # avoid unnecessary iterations below
        if not isinstance(cls, str) and hasattr(cls, "__iter__"):
            # we got some form of container which allows iteration, this can be tricky, see below
            if hasattr(cls, "__setitem__"):
                # if setitem is supported, we can simply iterate over the elements and replace them
                for i, c in list(enumerate(cls)):  # list to allow modification of the container on the go
                    c = make_picklable(c)
                    cls[i] = c
            else:
                # if there is an __iter__ but no __setitem__, we have to hope it is not a class, but a basic container
                # like a tuple or set. Then we can simply exchange the whole container, otherwise we cannot
                # make the elements picklable, and we have to hope that the class does not contain any non-picklable
                # elements.
                # To pinpoint the failure and to make it understandable which class is lacking a __setitem__ method,
                # we give a warning.
                # If one class should not have a __setitem__ method, or is not worth to have a __setitem__ method,
                # but also never contains non-picklable elements, we add it to the white_list above
                try:
                    new_items = []
                    for c in cls:
                        new_items.append(make_picklable(c))
                    if isinstance(cls, tuple):
                        cls = tuple(new_items)  # type: ignore
                    elif isinstance(cls, set):
                        cls = set(new_items)  # type: ignore
                    elif not isinstance(cls, type(new_items)):
                        # dev-note: if you want to remove this warning for a specific class and you are not sure
                        # where the class is defined, the following code might help you find it:
                        # from inspect import getsourcefile, getdoc
                        # print(getdoc(cls))
                        # print(getdoc(cls.__class__))
                        # print(getsourcefile(cls))
                        # print(getsourcefile(cls.__class__))
                        warnings.warn(f"{cls.__class__.__name__} does support __iter__ but not __setitem__, and is not "
                                      f"a tuple or list, this might cause that, we are not able to make it picklable.")
                    else:
                        cls = new_items  # type: ignore
                except UnsupportedOperation:
                    # this can occur if we try to loop over a file-like object, which is not supported,
                    # although it supports __iter__
                    pass
        if hasattr(cls, "unset_collections"):
            cls.unset_collections()
        if hasattr(cls, "__dict__"):
            # class supports dictionary to loop over members
            cls = item_loop(cls.__dict__.items(), cls)
        if hasattr(cls, "__slots__"):
            # class wants to be efficient and does not use a dictionary, but a list of slots
            # however slots can also be only a single string, so we have to check for that
            if isinstance(cls.__slots__, str):
                attr = {cls.__slots__: getattr(cls, cls.__slots__, None)}
            else:
                attr = {k: getattr(cls, k, None) for k in cls.__slots__}
            cls = item_loop(attr.items(), cls)
    return cls


class ReadAble(ABC):
    """
    A class to define an object from which data can be read.
    This class has on purpose identical names to some methods of the Connection class, but does not inherit from it.
    """

    @abstractmethod
    def poll(self) -> bool:
        """
        Returns whether new information can be read.
        """

    @abstractmethod
    def recv(self) -> Any:
        """
        Read new information from the object.

        Returns
        -------
        Any
            The read information.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Close the information connection.
        """

    @abstractmethod
    def was_closed(self) -> bool:
        """
        Whether the connection was closed.
        """


class StopReading:
    """
    An object that can be put into an information connection to signal that the connection should be closed.
    """


class ClosedConnectionException(Exception):
    """
    Exception that is thrown if we try to read information from a closed connection.
    """


def read_connection(connection: Union[Connection, ReadAble, None], return_first_signal: bool = False,
                    wanted_type: Optional[Type[WantedType]] = None) -> Optional[WantedType]:
    """
    Read information from a connection, that can be either the standard Connection class or an implementation of
    the ReadAble class.

    Parameters
    ----------
    connection : Union[Connection, ReadAble, None]
        The object to read the information from.
    return_first_signal : bool, optional
        If the very first object in the connection should be returned or whether we should as long as
        there is information in the connection, by default False
    wanted_type : Type[WantedType], optional
        The type of the object that is expected to be read, by default Any (no check is carried out)
        If the check is carried out and an incorrect type is read, a ValueError is raised, unless the value is None.
        Subscripted generics cannot be used because Python's typing is incomplete.

    Returns
    -------
    Optional[WantedType]
        The read information or None if the connection is None or an error occurred.
    """
    if connection is None:
        return None
    result = None
    while connection.poll():
        try:
            result = connection.recv()
            if return_first_signal:
                break
        except (EOFError, ClosedConnectionException):
            break
    if wanted_type is not None and result is not None and not isinstance(result, wanted_type):
        raise ValueError(f"Expected to read an object of type {wanted_type}, but received {type(result)}")
    return result


class MultiProcessingConnectionsWithProxyThread(ReadAble):
    """
    This class is a wrapper around the Connection class, that allows to read information from the connection
    while simultaneously reading information from it and saving it in memory via a thread which avoids blocking
    the connection with too large information packages.
    """

    def __init__(self, connection: Connection) -> None:
        """
        Constructs the class with an existing connection.

        Parameters
        ----------
        connection : Connection
            The existing connection, it must be readable.
        """
        self._close_event = Event()
        self._join_event = Event()
        self._memory: List[Any] = []
        self._proxy_thread = Thread(target=self._listen_proxy, args=(connection, self._close_event), daemon=True)
        self._proxy_thread.start()

    @classmethod
    def construct_connections(cls) -> Tuple[Any, Connection]:
        """
        Constructs the class with a new pair of read and write connections with this class as the read connection.

        Returns
        -------
        Tuple[Any, Connection]
            This class as the read connection and a new write connection.
        """
        recv, send = Pipe(duplex=False)
        return cls(recv), send

    def poll(self) -> bool:
        """
        Returns whether new information can be read.
        """
        return len(self._memory) > 0

    def recv(self) -> Any:
        """
        Returns new information.

        Raises
        ------
        ClosedConnectionException
            If our thread has stopped and we don't have any information left.
        """
        while not self._memory:
            if not self._proxy_thread.is_alive():
                raise ClosedConnectionException("Proxy thread was stopped and therefore connection was closed")
        return self._memory.pop(0)

    def close(self):
        """
        Closes the connection and stops the thread.
        """
        if self._close_event.is_set():
            self._join()
            return
        self._close_event.set()
        while self._proxy_thread.is_alive():
            pass
        self._join()

    def _join(self):
        """
        Joins the thread.
        """
        if not self._join_event.is_set():
            self._join_event.set()
            self._proxy_thread.join()

    def __del__(self):
        """
        If we have not been closed, we try to cleanup the connection.
        """
        try:
            self.close()
        except BaseException:
            pass

    def was_closed(self) -> bool:
        """
        Whether the connection was closed.
        """
        return self._close_event.is_set()

    def _listen_proxy(self, connection: Connection, close_event: Event):
        """
        A continuous method that is run in a thread and reads information from the connection and saves it in memory.

        Parameters
        ----------
        connection : Connection
            The connection to read from.
        close_event : Event
            A threadsafe event that signals that the thread should stop.
        """
        while not close_event.is_set():
            obj = read_connection(connection, return_first_signal=True)
            if isinstance(obj, StopReading):
                close_event.set()
            elif obj is not None:
                self._memory.append(obj)
            sleep(0.01)  # decrease cpu load
