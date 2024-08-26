#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import argparse
from pathlib import Path
from typing import List

import scine_utilities as utils
from scine_chemoton.utilities.reactive_complexes.inter_reactive_complexes import (
    InterReactiveComplexes,
    assemble_reactive_complex
)


from ._version import __version__  # noqa: F401


def create_reactive_complex_cli() -> None:
    """
    Easily assemble reactive complexes from two molecules and write them to rc.<n>.xyz files in the current directory
    from the command line.
    """

    args = _get_arguments()
    molecule1 = _read_file_to_molecule(args.molecule1)
    molecule2 = _read_file_to_molecule(args.molecule2)
    _create_reactive_complexes(molecule1, molecule2,
                               args.lhs, args.rhs, args.n_rotamers, args.multiple_attack_points, args.verbose)


def _get_arguments(argv=None) -> argparse.Namespace:
    """
    Parse the command line arguments to an argparse.Namespace object.

    Parameters
    ----------
    argv : _type_, optional
        By default None, introduced for testing

    Returns
    -------
     : argparse.Namespace
        The arguments parsed from the command line as argparse.Namespace object.
    """

    parser = argparse.ArgumentParser(
        description="Create reactive complexes from two molecules and write them to rc.<n>.xyz files.")
    parser.add_argument('-m1', '--molecule1', type=Path, required=True)
    parser.add_argument('-m2', '--molecule2', type=Path, required=True)
    parser.add_argument('-l', '--lhs', required=True, nargs="+", type=int,
                        help="The indices of the atoms of the first molecule that define the reaction coordinates")
    parser.add_argument('-r', '--rhs', required=True, nargs="+", type=int,
                        help="The indices of the atoms of the second molecule that define the reaction coordinates")
    # Not required
    parser.add_argument('-nrot', '--n_rotamers', required=False, type=int, default=2,
                        help="The number of rotamers to generate for each reactive complex")
    parser.add_argument('-ma', '--multiple_attack_points', required=False, action='store_true', default=False,
                        help="Whether to consider multiple attack points for each reactive complex")
    parser.add_argument('-v', '--verbose', required=False, action='store_true', default=False,
                        help="Whether to print verbose output")

    args = parser.parse_args(argv)

    return args


def _read_file_to_molecule(filepath: Path) -> utils.AtomCollection:
    """
    Read a file to an AtomCollection.

    Parameters
    ----------
    filepath : Path
        The path to the file to be read.

    Returns
    -------
    molecule : utils.AtomCollection
        The molecule read from the file as AtomCollection.
    """
    ending = filepath.suffix
    if ending == ".xyz":
        molecule = utils.io.read(str(filepath.expanduser()))[0]
    elif ending == ".mol":
        molecule = utils.io.read(str(filepath.expanduser()))[0]
    else:
        raise RuntimeError(f"Unsupported file type for AtomCollection {ending}")
    return molecule


def _create_reactive_complexes(molecule1: utils.AtomCollection, molecule2: utils.AtomCollection,
                               lhs: List[int], rhs: List[int],
                               n_rotamers: int, multiple_attack_points: bool, verbose: bool) -> None:
    """
    Create reactive complexes from two molecules and write them to rc.<n>.xyz files.

    Parameters
    ----------
    molecule1 : utils.AtomCollection
        First molecule
    molecule2 : utils.AtomCollection
        Second molecule
    lhs : List[int]
        Indices of the atoms of the first molecule that define the reaction coordinates
    rhs : List[int]
        Indices of the atoms of the second molecule that define the reaction coordinates
    n_rotamers : int
        Number of rotamers to generate for each reactive complex
    multiple_attack_points : bool
        Whether to consider multiple attack points for each reactive complex
    verbose : bool
        Whether to print verbose output.

    Writes
    ------
    rc.<n>.xyz : utils.AtomCollection
        The reactive complexes as AtomCollection written to files.
    """
    trial_generator = InterReactiveComplexes()
    trial_generator.options.number_rotamers = n_rotamers
    trial_generator.options.multiple_attack_points = multiple_attack_points

    reactive_axes = []
    for l_index, r_index in zip(lhs, rhs):
        reactive_axes.append((l_index, r_index))
    count = 0
    for (inter_coord, align1, align2, rot, spread, ) in trial_generator._generate_reactive_complexes(
            molecule1, molecule2, "", "", [reactive_axes]):

        lhs_atoms = [i[0] for i in inter_coord]
        rhs_atoms = [i[1] for i in inter_coord]
        if verbose:
            print("NT Ass. in RC:", [(i[0], i[1] + molecule1.size()) for i in inter_coord])
            print("Align1:\n" + "[" + ", ".join([f"{i:.9f}" for i in align1]) + "]")
            print("Align2:\n" + "[" + ", ".join([f"{i:.9f}" for i in align2]) + "]")
            print("Rot.:", rot)
            print("Spread:", spread)
            print(inter_coord, align1, align2, rot, spread)
        rc = assemble_reactive_complex(molecule1, molecule2,
                                       lhs_atoms, rhs_atoms, list(align1), list(align2), rot, spread)
        utils.io.write("rc." + str(count) + ".xyz", rc[0])
        count += 1
