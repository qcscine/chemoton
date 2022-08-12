#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from copy import deepcopy
import os
import numpy as np

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application tests imports
from ... import test_database_setup as db_setup
from ...resources import resources_root_path

# Local application imports
from ....utilities.reactive_complexes.lebedev_sphere import LebedevSphere
from ....utilities.reactive_complexes.unit_circle import UnitCircle
from ....utilities.reactive_complexes.inter_reactive_complexes import (
    InterReactiveComplexes,
    assemble_reactive_complex,
)


def test_class_init_and_options():
    reactive_complex = InterReactiveComplexes()

    assert reactive_complex.options.number_rotamers == 2
    assert reactive_complex.options.number_rotamers_two_on_two == 1
    assert reactive_complex.options.multiple_attack_points


def test_rotation_to_vector():

    start = np.array([2.0, 0.0, 0.0])
    target = np.array([0.5, 0.5, 0.5])

    r = InterReactiveComplexes._rotation_to_vector(start, target)
    end = r.T.dot(start)

    # end and target are parallel
    assert np.linalg.norm(np.cross(end, target)) < 1e-12
    # normalized vectors are identical
    assert abs(target / np.linalg.norm(target) - end / np.linalg.norm(end)).all() < 1e-12


def test_calculate_x_shift():
    # # # Load water
    rr = resources_root_path()
    water = utils.io.read(os.path.join(rr, "water.xyz"))[0]
    water_coords = water.positions
    hydrogen = utils.io.read(os.path.join(rr, "hydrogen.xyz"))[0]
    hydrogen_coords = hydrogen.positions

    x_shift = InterReactiveComplexes._calculate_x_shift(
        water_coords, water.elements, hydrogen_coords, hydrogen.elements
    )
    x_shift_v = np.array([x_shift, 0, 0])
    # # # Shift water in - x-direction
    water.positions = water_coords - x_shift_v
    # # # Shift hydrogen in + x-direction
    hydrogen.positions = hydrogen_coords + x_shift_v
    # # #  Min distance is twice vdw radius of hydrogen
    min_dist = 2 * utils.ElementInfo.vdw_radius(hydrogen.elements[0])

    for pos in water.positions:
        dist = np.linalg.norm(pos - hydrogen.positions[0])
        assert (dist - min_dist) >= 0.0


def test_prune_buried_points():

    # # # Get LebedevSphere
    lebedev = LebedevSphere()
    initial_points = lebedev.points
    # # # Load water
    rr = resources_root_path()
    water = utils.io.read(os.path.join(rr, "water.xyz"))[0]
    coords = water.positions
    elements = water.elements
    # # # Index of oxygen atom
    index = 2
    # # # Move sphere around oxygen atom
    possible_directions = deepcopy(initial_points) * utils.ElementInfo.vdw_radius(elements[index])
    possible_directions += coords[index]
    # # # Test for no pruning, if VdW scale is set to 0
    no_pruning = InterReactiveComplexes._prune_buried_points([2], coords, elements, possible_directions, 0.0)
    assert len(no_pruning) == len(possible_directions)
    # # # Test for purning for oxygen atom in water
    pruned_directions = InterReactiveComplexes._prune_buried_points([2], coords, elements, possible_directions)
    assert len(pruned_directions) == 5226


def test_prune_close_attack_points():

    radius = 2.0
    positions = np.array(
        [
            [0, radius, 1.0],
            [radius * np.cos(75 * utils.PI / 180), radius * np.sin(75 * utils.PI / 180), 1.0],
            [radius * np.cos(-75 * utils.PI / 180), radius * np.sin(-75 * utils.PI / 180), 1.0],
        ]
    )
    repulsion = np.array([1e-6, 1e-6, 10])
    # # # Check with default of 20 degrees, should return two points
    pruned_positions = InterReactiveComplexes._prune_close_attack_points(positions, repulsion, radius)

    assert len(pruned_positions) == 2
    assert np.linalg.norm(positions[2] - pruned_positions[-1]) < 1e-6

    # # # Check with 10 degrees, should return all points
    pruned_positions = InterReactiveComplexes._prune_close_attack_points(positions, repulsion, radius, 10)

    assert len(pruned_positions) == 3


def test_prune_by_repulsion_atom():
    # # # Get LebedevSphere
    lebedev = LebedevSphere()
    initial_points = lebedev.points
    nearest_neighbors = lebedev.nearest_neighbors
    # # # Load water
    rr = resources_root_path()
    water = utils.io.read(os.path.join(rr, "water.xyz"))[0]
    coords = water.positions
    elements = water.elements
    # # # Index of oxygen atom
    index = 2
    for index in [2, 0]:
        radius = utils.ElementInfo.vdw_radius(elements[index])
        # # # Move sphere around oxygen atom
        possible_directions = deepcopy(initial_points) * utils.ElementInfo.vdw_radius(elements[index])
        possible_directions += coords[index]

        reactive_complex = InterReactiveComplexes()
        # Default pruning
        pruned_directions = reactive_complex._prune_by_repulsion(
            [index], coords, elements, possible_directions, nearest_neighbors, radius
        )
        # Test oxygen, default and 180
        # Default angle must return two points
        if index == 2:
            assert len(pruned_directions) == 2
            # Pruning with angle set to 180
            pruned_directions = reactive_complex._prune_by_repulsion(
                [index], coords, elements, possible_directions, nearest_neighbors, radius, 180
            )

            # Angle of 180 must return only one point, with lower repulsion
            ref_direction = np.array([-4.50137903e-03, -3.61081183e00, -1.68185625e-08])
            assert len(pruned_directions) == 1
            assert all([abs(pruned - ref) < 1e-6 for pruned, ref in zip(pruned_directions[0], ref_direction)])
        # Test hydrogen
        if index == 0:
            ref_direction = np.array([-3.36993749, 1.35777718, 0.05821183])
            assert len(pruned_directions) == 1
            assert all([abs(pruned - ref) < 1e-6 for pruned, ref in zip(pruned_directions[0], ref_direction)])


def test_prune_by_repulsion_bond():
    # # # Get UnitCircle
    unit_circle = UnitCircle()
    initial_points = np.append(unit_circle.points, np.zeros((100, 1)), axis=1)
    nearest_neighbors = unit_circle.nearest_neighbors
    # # # Load methane
    rr = resources_root_path()
    methane = utils.io.read(os.path.join(rr, "methane.xyz"))[0]
    coords = methane.positions
    elements = methane.elements
    # # # Indices for bond
    for indices in [[0, 1], [1, 2]]:
        radius = 0.5 * (
            utils.ElementInfo.vdw_radius(elements[indices[0]]) + utils.ElementInfo.vdw_radius(elements[indices[1]])
        )

        # # # Move circle around center of bond
        possible_directions = deepcopy(initial_points) * radius
        # Rotate circle s.t. its normal aligns with the interatom axis
        interatom = coords[indices[0]] - coords[indices[1]]  # Interatom axis
        circle_normal = np.array([0.0, 0.0, 1.0])  # Initial points are in xy-plane
        r = InterReactiveComplexes._rotation_to_vector(circle_normal, interatom)
        possible_directions = (r.T.dot(possible_directions.T)).T
        # Move exactly between atoms
        possible_directions += 0.5 * (coords[indices[0]] + coords[indices[1]])

        reactive_complex = InterReactiveComplexes()
        pruned_directions = reactive_complex._prune_by_repulsion(
            indices, coords, elements, possible_directions, nearest_neighbors, radius
        )

        if indices == [0, 1]:
            ref_direction = np.array([3.02730346, 1.77840469, 1.74108634])
            assert len(pruned_directions) == 3
            assert all([abs(pruned - ref) < 1e-6 for pruned, ref in zip(pruned_directions[0], ref_direction)])

            pruned_directions = reactive_complex._prune_by_repulsion(
                indices, coords, elements, possible_directions, nearest_neighbors, radius, 120
            )
            assert len(pruned_directions) == 1

        if indices == [1, 2]:
            ref_direction = np.array([3.8721789, 2.36053838, -0.94589071])
            assert len(pruned_directions) == 3
            assert all([abs(pruned - ref) < 1e-6 for pruned, ref in zip(pruned_directions[0], ref_direction)])

            pruned_directions = reactive_complex._prune_by_repulsion(
                indices, coords, elements, possible_directions, nearest_neighbors, radius, 40
            )
            assert len(pruned_directions) == 2


def test_get_attack_points_per_atom():
    ref_attack_points = np.array(
        [
            [-3.36993749e00, 1.35777718e00, 5.82118267e-02],
            [3.37443887e00, 1.35459166e00, 5.82118267e-02],
            [-4.50137903e-03, 2.13395559e00, -1.68185625e-08],
            [-4.50137903e-03, -3.61081183e00, -1.68185625e-08],
        ]
    )
    # # # Load water
    rr = resources_root_path()
    water = utils.io.read(os.path.join(rr, "water.xyz"))[0]
    coords = water.positions
    elements = water.elements

    for index in [None, [0]]:
        # Get attack points for all atoms
        reactive_complex = InterReactiveComplexes()
        attack_points = reactive_complex._get_attack_points_per_atom(coords, elements, indices=index)
        # Flatten attack points
        flat_attack_points = []
        for pi in attack_points.values():
            for t in pi:
                flat_attack_points.append(t)
        flat_attack_points = np.asarray(flat_attack_points)

        if index is None:
            # Four attack points for this water
            assert len(flat_attack_points) == 4
            # Compare all coordinates with reference and guarantee that no element deviates more than 1e-6
            assert np.allclose(ref_attack_points, flat_attack_points, rtol=0, atol=1e-6)
        if index == [0]:
            assert len(flat_attack_points) == 1
            assert np.allclose(ref_attack_points[0], flat_attack_points, rtol=0, atol=1e-6)


def test_get_attack_points_per_atom_pair():
    ref_attack_points = np.array(
        [
            [3.02730346, 0.35068958, -2.65296896],
            [3.02730346, 1.77840469, 1.74108634],
            [3.02730346, -2.65042374, 0.7511259],
            [3.87243229, 2.36835156, -0.92345448],
        ]
    )
    # # # Load methane
    rr = resources_root_path()
    methane = utils.io.read(os.path.join(rr, "methane.xyz"))[0]
    coords = methane.positions
    elements = methane.elements

    valid_pairs = [(0, 1), (1, 2)]

    reactive_complexes = InterReactiveComplexes()
    attack_points = reactive_complexes._get_attack_points_per_atom_pair(coords, elements, valid_pairs)

    # Flatten attack points
    flat_attack_points = []
    for pi in attack_points.values():
        for t in pi:
            flat_attack_points.append(t)
    flat_attack_points = np.asarray(flat_attack_points)

    assert len(flat_attack_points) == 4
    # Compare all coordinates with reference and guarantee that no element deviates more than 1e-6
    assert np.allclose(ref_attack_points, flat_attack_points, rtol=0, atol=1e-6)


def test_set_up_rotamers_atom_on_atom():
    ref_op = (
        np.array([-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, -0.0, -1.0]),
        np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        0.0,
        3.0424590619871035,
    )
    # # # Load methane
    rr = resources_root_path()
    methane = utils.io.read(os.path.join(rr, "methane.xyz"))[0]
    coords1 = methane.positions
    elements1 = methane.elements
    # # # Load water
    rr = resources_root_path()
    water = utils.io.read(os.path.join(rr, "water.xyz"))[0]
    coords2 = water.positions
    elements2 = water.elements

    reactive_complex = InterReactiveComplexes()
    reactive_complex.options.multiple_attack_points = False
    reactive_complex.options.number_rotamers = 1
    reactive_complex.options.number_rotamers_two_on_two = 500

    # Attack points of C in methane
    attack_points1 = reactive_complex._get_attack_points_per_atom(coords1, elements1)
    index1 = (0,)
    # Attack points of O in oxygen
    attack_points2 = reactive_complex._get_attack_points_per_atom(coords2, elements2)
    index2 = (2,)
    # Get operations for setting up one rotamer
    operations = reactive_complex._set_up_rotamers(
        coords1,
        elements1,
        list(index1),
        attack_points1[index1][0:1],
        coords2,
        elements2,
        list(index2),
        attack_points2[index2][0:1],
    )
    print(operations)
    # Check, if one rotamer was generated
    assert len(operations) == 1
    tmp_op = operations[-1]
    # Check with ref operation
    for i in range(0, 4):
        assert np.allclose(tmp_op[i], ref_op[i], rtol=0, atol=1e-6)

    # Get operations for setting up three rotamers
    reactive_complex.options.number_rotamers = 3
    reactive_complex.options.number_rotamers_two_on_two = 500
    operations = reactive_complex._set_up_rotamers(
        coords1,
        elements1,
        list(index1),
        attack_points1[index1][0:1],
        coords2,
        elements2,
        list(index2),
        attack_points2[index2][0:1],
    )
    # Check, if three rotamers were generated
    assert len(operations) == 3
    # Check, if correct angles were generated
    tmp_angle = 0.0
    for tmp_op in operations:
        assert abs(tmp_op[2] * 180 / utils.PI - tmp_angle) < 1e-6
        tmp_angle += 360.0 / reactive_complex.options.number_rotamers


def test_set_up_rotamers_atom_on_pair():
    ref_op = (
        np.array([-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, -0.0, -0.0, -1.0]),
        np.array(
            [-0.57538376, -0.81774299, 0.01516377, 0.81774299, -0.57484223, 0.029203, -0.01516377, 0.029203, 0.99945848]
        ),
        0.0,
        3.0042740250843085,
    )
    # # # Load methane
    rr = resources_root_path()
    methane = utils.io.read(os.path.join(rr, "methane.xyz"))[0]
    coords1 = methane.positions
    elements1 = methane.elements
    # # # Load water
    rr = resources_root_path()
    water = utils.io.read(os.path.join(rr, "water.xyz"))[0]
    coords2 = water.positions
    elements2 = water.elements

    reactive_complex = InterReactiveComplexes()
    reactive_complex.options.multiple_attack_points = False
    reactive_complex.options.number_rotamers = 1
    reactive_complex.options.number_rotamers_two_on_two = 500

    # Attack points of C in methane
    attack_points1 = reactive_complex._get_attack_points_per_atom(coords1, elements1)
    index1 = (0,)
    # Attack points of OH bond in water
    attack_points2 = reactive_complex._get_attack_points_per_atom_pair(coords2, elements2, [(1, 2)])
    index2 = (1, 2)

    operations = reactive_complex._set_up_rotamers(
        coords1,
        elements1,
        list(index1),
        attack_points1[index1][0:1],
        coords2,
        elements2,
        list(index2),
        attack_points2[index2][0:1],
    )

    # Check, if one rotamer was generated
    assert len(operations) == 1
    tmp_op = operations[-1]
    # Check with ref operation
    for i in range(0, 4):
        assert np.allclose(tmp_op[i], ref_op[i], rtol=0, atol=1e-6)

    # Get operations for setting up four rotamers
    reactive_complex.options.number_rotamers = 4
    reactive_complex.options.number_rotamers_two_on_two = 4
    operations = reactive_complex._set_up_rotamers(
        coords1,
        elements1,
        list(index1),
        attack_points1[index1][0:1],
        coords2,
        elements2,
        list(index2),
        attack_points2[index2][0:1],
    )
    # Check, if three rotamers were generated
    assert len(operations) == 4
    # Check, if correct angles were generated
    tmp_angle = 0.0
    for tmp_op in operations:
        assert abs(tmp_op[2] * 180 / utils.PI - tmp_angle) < 1e-6
        tmp_angle += 360.0 / reactive_complex.options.number_rotamers


def test_set_up_rotamers_pair_on_pair():
    ref_op = (
        np.array(
            [
                0.57538376,
                0.81774299,
                -0.01516377,
                -0.81774299,
                0.57552972,
                0.00787114,
                0.01516377,
                0.00787114,
                0.99985404,
            ]
        ),
        np.array(
            [
                -0.57538376,
                0.81356599,
                0.08392803,
                0.81774299,
                0.57415048,
                0.04059111,
                -0.01516377,
                0.09198703,
                -0.99564474,
            ]
        ),
        0.0,
        2.8719820956426396,
    )
    # # # Load water
    rr = resources_root_path()
    water = utils.io.read(os.path.join(rr, "water.xyz"))[0]
    coords = water.positions
    elements = water.elements

    reactive_complex = InterReactiveComplexes()
    reactive_complex.options.multiple_attack_points = False
    reactive_complex.options.number_rotamers = 500
    reactive_complex.options.number_rotamers_two_on_two = 1

    # Attack points of OH bond in water
    pair = (1, 2)
    attack_points = reactive_complex._get_attack_points_per_atom_pair(coords, elements, [pair])

    # Check whether rotamers of complexes with 4 distinct atoms are properly aligned
    reactive_complex.options.number_rotamers = 500
    reactive_complex.options.number_rotamers_two_on_two = 1

    # Align pair[0] with pair[0] and pair[1] with pair[1]
    operations = reactive_complex._set_up_rotamers(
        coords, elements, list(pair), attack_points[pair][0:1], coords, elements, list(pair), attack_points[pair][0:1]
    )

    # Check, if one rotamer was generated
    assert len(operations) == 1
    tmp_op = operations[-1]
    # Check with ref operation
    for i in range(0, 4):
        assert np.allclose(tmp_op[i], ref_op[i], rtol=0, atol=1e-6)

    # Align pair[0] with pair[1] and pair[1] with pair[0] and generate one rotamer within 90 degrees
    reactive_complex.options.number_rotamers_two_on_two = 2
    operations += reactive_complex._set_up_rotamers(
        coords,
        elements,
        list(pair)[::-1],
        attack_points[pair][0:1],
        coords,
        elements,
        list(pair),
        attack_points[pair][0:1],
    )
    # Check whether two rotamers were generated
    assert len(operations) == 3

    # First complex should have a angle of 0.1 rad between the two OH bonds
    tmp_op = operations[0]
    react_c = assemble_reactive_complex(
        water, water, list(pair), list(pair), tmp_op[0], tmp_op[1], tmp_op[2], tmp_op[3]
    )[0]
    # Get OH vectors
    v1 = react_c.positions[2] - react_c.positions[1]
    v2 = react_c.positions[5] - react_c.positions[4]
    # Calculate angle between two vectors
    angle = np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    # Should be close to 0.1 (parallel + small distortion)
    assert abs(0.1 - angle) < 1e-6

    # Second and third complex should have antiparallel and orthogonal OH bonds
    angles = []
    tmp_op = operations[1]
    react_c = assemble_reactive_complex(
        water, water, list(pair)[::-1], list(pair), tmp_op[0], tmp_op[1], tmp_op[2], tmp_op[3]
    )[0]
    # Get OH vectors
    v1 = react_c.positions[2] - react_c.positions[1]
    v2 = react_c.positions[5] - react_c.positions[4]
    # Calculate angle between two vectors
    angles.append(np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    # Should be close to pi - 0.1 (antiparallel + small distortion)
    tmp_op = operations[2]
    react_c = assemble_reactive_complex(
        water, water, list(pair)[::-1], list(pair), tmp_op[0], tmp_op[1], tmp_op[2], tmp_op[3]
    )[0]
    # Get OH vectors
    v1 = react_c.positions[2] - react_c.positions[1]
    v2 = react_c.positions[5] - react_c.positions[4]
    # Calculate angle between two vectors
    angles.append(np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    angles.sort()
    # Smaller angle should be 90 degrees/0.5 pi rad  plus small distortion of 0.1 rad
    assert abs(0.5 * utils.PI + 0.1 - angles[0]) < 1e-6
    # Larger angle should be 180 degrees/pi rad minus small distortion of 0,1 rad
    assert abs(utils.PI - 0.1 - angles[1]) < 1e-6


def test_iterate_reactive_complexes():
    # Connect to test DB
    manager = db_setup.get_clean_db("chemoton_test_iterate_reactive_complexes")

    # Get collections
    structures = manager.get_collection("structures")

    rr = resources_root_path()

    # # # Load methane
    methane = db.Structure()
    methane.link(structures)
    methane.create(os.path.join(rr, "methane.xyz"), 0, 1)
    # # # Load water
    water = db.Structure()
    water.link(structures)
    water.create(os.path.join(rr, "water.xyz"), 0, 1)

    reactive_complex = InterReactiveComplexes()
    reactive_complex.options.multiple_attack_points = False
    reactive_complex.options.number_rotamers = 1

    # Check atom on atom
    rc_list = []
    # Combine atom 0 and 1 of methane with 1 and 2 of water, which should yield 4 complexes
    trial_reaction_coords = [((0, 1),), ((0, 2),), ((1, 1),), ((1, 2),)]
    for op in reactive_complex.generate_reactive_complexes(methane, water, trial_reaction_coords):
        rc_list.append(op)
    assert len(rc_list) == 4

    # Check "bond on bond"
    rc_list = []
    # Combine atom 0 of methane with 1 of water and 1 of methane with 2 of water simultaneously
    trial_reaction_coords = [((0, 1), (1, 2))]
    for op in reactive_complex.generate_reactive_complexes(methane, water, trial_reaction_coords):
        rc_list.append(op)
    assert len(rc_list) == 1

    # Check opposite alignment
    rc_list = []
    # Combine atom 0 of methane with 1 of water and 1 of methane with 2 of water simultaneously
    trial_reaction_coords = [((0, 2), (1, 1))]
    for op in reactive_complex.generate_reactive_complexes(methane, water, trial_reaction_coords):
        rc_list.append(op)
    assert len(rc_list) == 1

    # Check atom on bond and bond on bond
    trial_reaction_coords = [((0, 1), (0, 2)), ((0, 1), (1, 2))]
    rc_list = []
    for op in reactive_complex.generate_reactive_complexes(methane, water, trial_reaction_coords):
        rc_list.append(op)
    assert len(rc_list) == 2

    # Check atom on atom, atom on bond and bond on bond
    rc_list = []
    trial_reaction_coords = [((0, 2),), ((0, 1), (0, 2)), ((0, 2), (1, 2)), ((0, 1), (1, 2))]
    for op in reactive_complex.generate_reactive_complexes(methane, water, trial_reaction_coords):
        rc_list.append(op)
    assert len(rc_list) == 4

    # Check whether rotamer settings affect correct number of trial coordinates
    # This should affect all coordinates involving less than 4 distinct atoms i.e. 3 of the 4 trial coordinates
    reactive_complex.options.number_rotamers = 3
    rc_list = []
    for op in reactive_complex.generate_reactive_complexes(methane, water, trial_reaction_coords):
        rc_list.append(op)
    assert len(rc_list) == 10

    # Add one complex by also increasing the number of rotamers for the reactioncoordinate involving four distinct atoms
    reactive_complex.options.number_rotamers_two_on_two = 2
    rc_list = []
    for op in reactive_complex.generate_reactive_complexes(methane, water, trial_reaction_coords):
        rc_list.append(op)
    assert len(rc_list) == 11

    # Cleaning
    manager.wipe()


def test_pair_on_pair_generate_reactive_complexes():
    """
    Test whether the alignment of diatomic reactive fragments is
    conserved when complexes are generated with the generate_reactive_complexes
    method
    """
    manager = db_setup.get_clean_db("chemoton_test_iterate_reactive_complexes")
    structures = manager.get_collection("structures")
    rr = resources_root_path()
    water = db.Structure()
    water.link(structures)
    water.create(os.path.join(rr, "water.xyz"), 0, 1)
    water_atoms = water.get_atoms()

    # Test parallel and antiparallel alignment
    # first coordinate: H on H and O on O
    # second coordinte: H on O and O on H
    trial_reaction_coords = [((0, 0), (2, 2)), ((0, 2), (2, 0))]
    reactive_complex = InterReactiveComplexes()
    reactive_complex.options.multiple_attack_points = False
    reactive_complex.options.number_rotamers_two_on_two = 1

    rc_list = []
    for op in reactive_complex.generate_reactive_complexes(water, water, trial_reaction_coords):
        rc_list.append(op)
    assert len(rc_list) == 2

    # First complex should be with parallel O-H
    rc_op = rc_list[0]
    lhs = [pair[0] for pair in rc_op[0]]
    rhs = [pair[1] for pair in rc_op[0]]
    align_0 = rc_op[1]
    align_1 = rc_op[2]
    rot = rc_op[3]

    react_c = assemble_reactive_complex(water_atoms, water_atoms, lhs, rhs, align_0, align_1, rot)[0]

    v1 = react_c.positions[2] - react_c.positions[0]
    v2 = react_c.positions[5] - react_c.positions[3]
    # Calculate angle between two vectors
    angle = np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    # Should be close to 0.1 (parallel + small distortion)
    assert abs(0.1 - angle) < 1e-6

    # Second complex should be with antiparallel O-H
    rc_op = rc_list[1]
    lhs = [pair[0] for pair in rc_op[0]]
    rhs = [pair[1] for pair in rc_op[0]]
    align_0 = rc_op[1]
    align_1 = rc_op[2]
    rot = rc_op[3]

    react_c = assemble_reactive_complex(water_atoms, water_atoms, lhs, rhs, align_0, align_1, rot)[0]

    v1 = react_c.positions[2] - react_c.positions[0]
    v2 = react_c.positions[5] - react_c.positions[3]
    angle = np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    # 180 degrees - 0.1 rad distortion
    assert abs(utils.PI - 0.1 - angle) < 1e-6
    # Cleaning
    manager.wipe()
