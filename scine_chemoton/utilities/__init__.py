#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
from time import sleep
from typing import Callable, Tuple

import scine_database as db


def connect_to_db(credentials: db.Credentials) -> db.Manager:
    manager = db.Manager()
    manager.set_credentials(credentials)
    manager.connect()
    sleep(1.0)
    if not manager.has_collection("calculations"):
        raise RuntimeError(f"Database {credentials.database_name} is missing collections.")
    return manager


def yes_or_no_question(question: str, callable_input: Callable = input) -> bool:
    inp = callable_input(f"{question}? (y/n) ")
    while True:
        if inp.strip().lower() in ["y", "yes"]:
            return True
        if inp.strip().lower() in ["n", "no"]:
            return False
        inp = callable_input("Did not recognize answer, please answer 'yes' or 'no': ")


def integer_question(question: str, limits: Tuple[int, int], callable_input: Callable = input) -> int:
    inp = callable_input(f"{question}? ")
    while True:
        try:
            val = int(inp.strip())
            if limits[0] <= val <= limits[1]:
                return val
            inp = callable_input(f"Please enter an integer between {limits[0]} and {limits[1]}: ")
        except ValueError:
            inp = callable_input("Did not recognize answer, please give an integer: ")
