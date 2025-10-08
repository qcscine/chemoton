#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import numpy as np
from typing import List, Tuple, Dict, Union
from numpy.linalg import norm
from math import cos, sin, pi, sqrt
import random

import scine_utilities as utils
import scine_database as db
from pymatgen.core import IStructure
from pymatgen.core.surface import Slab
from polymethod import OverloadMeta, overload

from .property_names import surface_atom_indices_str


class PeriodicUtils(metaclass=OverloadMeta):

    @classmethod  # type: ignore
    @overload
    def to_periodic_system(cls, structure: db.Structure) -> utils.PeriodicSystem:  # noqa: F811
        model = structure.get_model()
        if not model.periodic_boundaries:
            raise RuntimeError("The given structure does not have periodic boundaries, so no PeriodicSystem can be "
                               "generated!")
        pbc = utils.PeriodicBoundaries(model.periodic_boundaries)
        ac = structure.get_atoms()
        return utils.PeriodicSystem(pbc, ac)

    @classmethod  # type: ignore
    @overload
    def to_periodic_system(cls, structure: db.Structure,  # noqa: F811
                           properties: db.Collection) -> utils.PeriodicSystem:
        if not structure.has_property(surface_atom_indices_str):
            raise RuntimeError("The given structure does not have 'surface_atom_indices' assigned")
        prop = db.VectorProperty(structure.get_property("surface_atom_indices"), properties)
        ps = cls.to_periodic_system(structure)
        ps.solid_state_atom_indices = set(prop.get_data())
        return ps

    @classmethod
    def clashing_adsorbates(cls, placed_adsorbates: np.ndarray, candidate: np.ndarray,
                            vdw_list: np.ndarray, pbc: utils.PeriodicBoundaries) -> bool:
        """
        checks whether any position of the candidate is clashing with any position from the already
        placed_adsorbates
        the corresponding vdw values have to be provided

        Parameters
        ----------
        placed_adsorbates : np.ndarray of shape (n,3)
            Atom positions.
        candidate : np.ndarray of shape (n,3)
            positions that could possibly clash
        vdw_list : np.ndarray
            The vdw values of all atoms of the adsorbate
        pbc : utils.PeriodicBoundaries
            The periodic boundaries

        Returns
        -------
        clashing : bool
            whether there is a clash
        """
        for i, pos in enumerate(candidate):
            for j, other_pos in enumerate(placed_adsorbates):
                if utils.geometry.distance(pos, other_pos, pbc) <= (vdw_list[i] + vdw_list[j % len(candidate)]):
                    return True
        return False

    @classmethod
    def periodic_clash(cls, adsorbate_atoms: utils.AtomCollection, ps: utils.PeriodicSystem):
        # now only checks clash between different adsorbates (previously included in ps)
        adsorb_vdw_list = np.asarray([utils.ElementInfo.covalent_radius(ele) for ele in adsorbate_atoms.elements])
        slab_vdw_list = np.asarray([utils.ElementInfo.covalent_radius(ele) for ele in ps.atoms.elements])
        for i, ad_atom in enumerate(adsorbate_atoms):
            for j, slab_atom in enumerate(ps.atoms):
                dist = utils.geometry.distance(ad_atom.position, slab_atom.position, ps.pbc)
                if dist <= (adsorb_vdw_list[i] + slab_vdw_list[j]):
                    return True
        return False

    @classmethod
    def true_adsorption_site(cls, site: np.ndarray, slab_atoms: utils.PeriodicSystem, vdw_reactive: float) -> bool:
        return cls.first_atom_hit_in_straight_line(site, slab_atoms.atoms, vdw_reactive) in \
            slab_atoms.solid_state_atom_indices

    @classmethod
    def first_atom_hit_in_straight_line(
            cls,
            site: np.ndarray,
            slab_atoms: utils.AtomCollection,
            vdw_reactive: float) -> int:
        hit_indices = []
        hit_z_positions = []
        vdw = [utils.ElementInfo.vdw_radius(ele) for ele in slab_atoms.elements]
        for i, pos in enumerate(slab_atoms.positions):
            if (pos[0] - site[0]) ** 2 + (pos[1] - site[1]) ** 2 < (vdw_reactive + vdw[i]) ** 2:
                hit_indices.append(i)
                hit_z_positions.append(pos[2])
        if len(hit_indices) == 0:
            raise RuntimeError('The specified adsorption site ' +
                               str(site) +
                               ' is not straight above the slab in the '
                               'z-direction '
                               ' with minx: ' +
                               str(np.min(slab_atoms.positions[:, 0])) +
                               ' with miny: ' +
                               str(np.min(slab_atoms.positions[:, 1])) +
                               ' with maxx: ' +
                               str(np.max(slab_atoms.positions[:, 0])) +
                               ' with maxy: ' +
                               str(np.max(slab_atoms.positions[:, 1])))
        return hit_indices[np.argmax(hit_z_positions)]

    @classmethod
    def translate_position_from_point_to_point(cls, positions: np.ndarray, start: np.ndarray, end: np.ndarray) -> \
            np.ndarray:
        """
        translate atoms with vector defined from start to end

        Parameters
        ----------
        positions : np.ndarray
            The positions of shape (n,3) to be translated
        start : np.ndarray of len 3
            The start point of the translation vector
        end : np.ndarray of len 3
            The end point of the translation vector

        Returns
        -------
        atoms : np.ndarray of shape (n, 3)
            the translated positions
        """
        trans_vec = end - start
        return np.add(trans_vec, positions)

    @classmethod
    def rotate_pos_around_vec(cls, positions: np.ndarray, vector: np.ndarray, origin: Union[np.ndarray, None] = None,
                              angle: Union[float, None] = None) -> np.ndarray:
        """
        rotate positions around a given vector
        optionally the base point of the vector and the angle of rotation can be given
        otherwise the Cartesian origin and a random angle are used

        Parameters
        ----------
        positions : np.ndarray
            The positions of shape (n,3) to be rotated
        vector : np.ndarray of len 3
            The vector around the rotation is performed
        origin : either a np.ndarray of len 3 or None
            The base point of the rotation vector
        angle : float or None
            The angle of rotation in radians

        Returns
        -------
        new_positions : np.ndarray
            the rotated positions
        """
        if origin is not None:
            positions = np.add(-1 * origin, positions)
        v = vector / norm(vector)
        rotM = np.zeros(shape=(3, 3))
        if angle is None:
            angle = np.random.uniform(0.0, 2 * pi)
        a = angle
        rotM[0][0] = cos(a) + v[0] ** 2 * (1 - cos(a))
        rotM[1][1] = cos(a) + v[1] ** 2 * (1 - cos(a))
        rotM[2][2] = cos(a) + v[2] ** 2 * (1 - cos(a))

        rotM[0][1] = v[0] * v[1] * (1 - cos(a)) - v[2] * sin(a)
        rotM[1][0] = v[0] * v[1] * (1 - cos(a)) + v[2] * sin(a)

        rotM[0][2] = v[0] * v[2] * (1 - cos(a)) + v[1] * sin(a)
        rotM[2][0] = v[0] * v[2] * (1 - cos(a)) - v[1] * sin(a)

        rotM[1][2] = v[1] * v[2] * (1 - cos(a)) - v[0] * sin(a)
        rotM[2][1] = v[1] * v[2] * (1 - cos(a)) + v[0] * sin(a)

        new_positions = (np.dot(rotM, positions.T)).T
        if origin is not None:
            new_positions = np.add(origin, new_positions)
        return new_positions

    @classmethod
    def get_closest_element(cls, atoms: utils.AtomCollection, pos: np.ndarray) -> Tuple[utils.ElementType, int]:
        """
        get element of the closest atom to given position

        Parameters
        ----------
        atoms : utils.AtomCollection
            The atoms from which we want to find the closest one
        pos : np.ndarray of len 3
            The position to which we want to find the closest atom

        Returns
        -------
        element : utils.ElementType
            the closest element
        min_index : int
            the index of the closest element
        """
        dists = np.zeros(len(atoms))
        for i in range(len(atoms)):
            dists[i] = norm(pos - atoms.get_position(i))
        min_index = int(np.argmin(dists))
        return atoms.get_element(min_index), min_index

    @classmethod
    def get_surface_energy(cls, crystal: IStructure, slab: Slab, equation: int = 3, unit: str = 'J/m2') -> float:
        """
        get empirical estimate of surface energy

        Parameters
        ----------
        crystal : pymatgen.core.IStructure
            The crystal from which the Slab was generated from
        slab : Slab
            The slab from which we want the surface energy
        equation : int
            The empirical formula used for the estimate
        unit : str
            The unit of the surface energy

        Returns
        -------
        gamma : float
            The surface energy
        """
        from . import cohesive_energies

        JOULE_PER_EV = 1.60218e-19
        SQUAREMETER_PER_SQUAREANGSTROM = 1e-20

        crystal_neighbor_dict = cls._get_crystal_neighbor_dict(crystal)
        slab_neighbor_list = cls.get_slab_neighbor_list(slab)

        surface_n = 0
        gamma = 0.0
        for site, n in zip(slab.sites, slab_neighbor_list):
            ele = site.species_string
            CN_bulk = crystal_neighbor_dict[ele]
            if n != CN_bulk:
                surface_n += 1
                if equation == 1:
                    gamma += cohesive_energies.elements[ele] * n / CN_bulk  # eq 1
                elif equation == 2:
                    gamma += cohesive_energies.elements[ele] * (CN_bulk - n) / CN_bulk  # eq 2
                elif equation == 3:
                    gamma += cohesive_energies.elements[ele] * (sqrt(CN_bulk) - sqrt(n)) / sqrt(CN_bulk)  # eq 3
                else:
                    raise NotImplementedError('Only equation 1, 2, and 3 are possible')
        # gamma units:
        gamma /= surface_n  # ev / atom
        if unit == 'ev/atom':
            return gamma
        gamma *= surface_n  # ev
        gamma /= 2 * slab.surface_area  # ev / A^2
        gamma *= JOULE_PER_EV / SQUAREMETER_PER_SQUAREANGSTROM  # J / m^2
        if unit == 'J/m2':
            return gamma
        elif unit == 'kJ/mol':
            gamma *= SQUAREMETER_PER_SQUAREANGSTROM  # J / A^2
            gamma *= 2 * slab.surface_area  # J
            gamma /= surface_n  # J / atom
            gamma *= 6.022e23 / 1000  # kJ / mol
            return gamma
        else:
            raise NotImplementedError("Only available units are 'J/m2', 'ev/atom', and 'kJ/mol'")

    @classmethod
    def _get_crystal_neighbor_dict(cls, crystal: IStructure) -> Dict[str, int]:
        """
        get dictionary of elements and number of neighbors within a crystal

        Parameters
        ----------
        crystal : pymatgen.core.IStructure
            The crystal

        Returns
        -------
        crystal_neighbor_dict : Dict[str, int]
            the dictionary containing the neighbor count for each element in the crystal
        """

        d = np.min(np.array([crystal.get_distance(0, i) for i in range(1, len(crystal.sites))]))
        nl = crystal.get_neighbor_list(d + 1e-6)
        crystal_neighbor_dict: Dict[str, int] = {}
        for i, site in enumerate(crystal.sites):
            ele = site.species_string
            if ele not in crystal_neighbor_dict.keys():
                crystal_neighbor_dict[ele] = np.count_nonzero(nl[0] == i)
        return crystal_neighbor_dict

    @classmethod
    def get_slab_neighbor_list(cls, slab: Slab) -> List[int]:
        """
        get list number of neighbors for each atom in a slab

        Parameters
        ----------
        slab : Slab
            The slab

        Returns
        -------
        slab_neighbor_dict : List[int]
            the list containing the neighbor count for each atom in the slab
        """
        d = np.min(np.array([slab.get_distance(0, i) for i in range(1, len(slab.sites))]))
        nl = slab.get_neighbor_list(d + 1e-6)
        return [np.count_nonzero(nl[0] == i) for i in range(len(slab.sites))]

    @classmethod
    def get_vac_surface(cls, crystal: IStructure, slab: Slab, concentration: float, max_extension: int = 10,
                        eps: float = 0.01) -> Slab:
        """
        get slab with vacancies randomly distributed based on given concentration of vacancies

        Parameters
        ----------
        crystal : pymatgen.core.IStructure
            The crystal from which the slab was cut from
        slab : Slab
            The slab
        concentration : float
            The concentration of vacancies, 0.0 corresponds to no vacancies, 1.0 corresponds to a missing surface layer
        max_extension : int
            The maximum allowed extension of the surface in x and y direction
        eps : float
            The tolerance for the achieved concentration

        Returns
        -------
        slab_mod : Slab
            the slab containing vacancies
        """
        n_surface = cls.get_n_surface_atoms(crystal, slab)
        extension, n_vac = cls._determine_extension_and_vac(n_surface, concentration, max_extension, eps)
        return cls.surface_vac_slab(slab, extension, n_vac)

    @classmethod
    def get_n_surface_atoms(cls, crystal: IStructure, slab: Slab) -> int:
        """
        get the number of surface atoms

        Parameters
        ----------
        crystal : pymatgen.core.IStructure
            The crystal from which the slab was cut from
        slab : Slab
            The slab

        Returns
        -------
        surface_n : int
            the number of surface atoms
        """
        crystal_neighbor_dict = cls._get_crystal_neighbor_dict(crystal)
        slab_neighbor_list = cls.get_slab_neighbor_list(slab)
        surface_n = 0
        for site, n in zip(slab.sites, slab_neighbor_list):
            ele = site.species_string
            CN_bulk = crystal_neighbor_dict[ele]
            if n != CN_bulk:
                surface_n += 1
        return surface_n

    @classmethod
    def _determine_extension_and_vac(cls, n_surface_atoms: int, wanted_concentration: float,
                                     max_extension: int, eps: float) -> Tuple[List[int], int]:
        """
        determine the needed extension and number of vacancies to achieve the wanted concentration of vacancies

        Parameters
        ----------
        n_surface_atoms : int
            The number of atoms on the surface on the non-extended surface
        wanted_concentration : float
            The concentration of vacancies, 0.0 corresponds to no vacancies, 1.0 corresponds to a missing surface layer
        max_extension : int
            The maximum allowed extension of the surface in x and y direction
        eps : float
            The tolerance for the achieved concentration

        Returns
        -------
        extension, n_vac : Tuple[List[int], int]
            the extension in each dimension and the number of vacancies
        """
        for i in range(max_extension):
            extension = i + 1
            all_sites = n_surface_atoms * extension ** 2
            possible_deltas = np.zeros(all_sites + 1)
            for j in range(all_sites + 1):
                concentration = j / all_sites
                possible_deltas[j] = abs(concentration - wanted_concentration)
            if np.min(possible_deltas) < eps:
                return [extension, extension, 1], int(np.argmin(possible_deltas))
        raise RuntimeError("Could not get right concentration within allowed extension of surface.")

    @classmethod
    def surface_vac_slab(cls, slab: Slab, extension: List[int], n_vac: int = 1,
                         along_x_axis: bool = False) -> Slab:
        """
        get slab with given extension and n given vacancies

        Parameters
        ----------
        slab : Slab
            The slab
        extension : Union[List[int], None]
            The extension of the Slab in each dimension
        n_vac : int
            The number of vacancies in the new slab
        along_x_axis : bool
            whether the vacancies shall be created along the x-axis or randomly (default)

        Returns
        -------
        slab_mod : Slab
            the extended slab with vacancies
        """
        superslab = slab * extension
        if along_x_axis:
            # leads to grooves, since argmin always returns first min -> always index 0
            slab_neighbor_list = cls.get_slab_neighbor_list(superslab)  # type: ignore
            remove_indices: List[int] = []
            original_remove_indices = []  # help list for right indices
            while len(remove_indices) < n_vac:
                index = int(np.argmin(slab_neighbor_list))
                del slab_neighbor_list[index]
                original_remove_indices.append(index)
                # index not applicable to total list, once indices have been removed
                # --> add for any lower indices that have already been removed
                index += sum(i <= index for i in original_remove_indices)
                remove_indices.append(index)
        else:
            min_neighbor_indices = cls.find_indices_of_min_neighbors(superslab)  # type: ignore
            remove_indices = random.sample(min_neighbor_indices, n_vac)
        ele = [s for i, s in enumerate(superslab.species) if i not in remove_indices]
        sites = [s for i, s in enumerate(superslab.sites) if i not in remove_indices]
        coords = np.array([site.coords for site in sites])
        slab_mod = Slab(superslab.lattice, ele, coords, slab.miller_index, slab.oriented_unit_cell, slab.shift,
                        slab.scale_factor, coords_are_cartesian=True)
        return slab_mod

    @classmethod
    def find_indices_of_min_neighbors(cls, slab: Slab) -> List[int]:
        """
        get indices of sites with the minimum number of neighbors

        Parameters
        ----------
        slab : Slab
            The slab

        Returns
        -------
        indices : List[int]
            the list of indices
        """
        slab_neighbor_list = cls.get_slab_neighbor_list(slab)
        min_value = None
        indices = []
        for ind, value in enumerate(slab_neighbor_list):
            if slab.frac_coords[ind][2] >= 0.5:
                if min_value is None or value == min_value:
                    min_value = value
                    indices.append(ind)
                elif value < min_value:
                    min_value = value
                    # overwrite existing list
                    indices = [ind]
        return indices

    @classmethod
    def get_top_surface_atoms_indices(cls, ps: utils.PeriodicSystem) -> List[int]:
        layers = cls.get_layer_indices(ps, relevant_indices=ps.solid_state_atom_indices)
        # inefficient because 2 iterations through list but easy solution
        max_layer = np.max(layers)
        return [i for i, layer in enumerate(layers) if layer == max_layer]

    @classmethod
    def get_n_top_surface_atoms(cls, ps: utils.PeriodicSystem) -> int:
        return len(cls.get_top_surface_atoms_indices(ps))

    @classmethod  # type: ignore
    @overload
    def is_surface(cls, compound: db.Compound, structures: db.Collection) -> bool:  # noqa: F811
        structure = db.Structure(compound.get_centroid(), structures)
        return cls.is_surface(structure)

    @classmethod  # type: ignore
    @overload
    def is_surface(cls, structure: db.Structure, structures: db.Collection) -> bool:  # noqa: F811
        structure.link(structures)
        return cls.is_surface(structure)

    @classmethod  # type: ignore
    @overload
    def is_surface(cls, structure: db.Structure) -> bool:  # noqa: F811
        return "surface" in str(structure.get_label()).lower()

    @classmethod
    def get_layer_indices(cls, ps: utils.PeriodicSystem, relevant_indices=None, eps: float = 0.5) -> List[int]:
        """
        Bin all atoms of the system into layers
        """
        rel_eps = eps / np.linalg.norm(ps.pbc.c)
        rel_all_coords = ps.pbc.transform(ps.atoms.positions, False)
        if relevant_indices is None:
            rel_coords = rel_all_coords
            relevant_indices = set(range(len(ps.atoms)))
        else:
            rel_coords = np.array([c for i, c in enumerate(rel_all_coords) if i in relevant_indices])
        # bin relative c coords
        bins = np.arange(np.min(rel_coords[:, 2]), np.max(rel_coords[:, 2]), rel_eps)
        digitized = np.digitize(rel_coords[:, 2], bins)
        # remove empty bin numbers
        new_digitized = digitized[:]
        for i in range(len(bins)):
            if i not in digitized:
                new_digitized = np.array([new_d - 1 if old_d > i else new_d
                                          for old_d, new_d in zip(digitized, new_digitized)])
        n_atoms = len(ps.atoms)
        all_layers_indices = -1 * np.ones(n_atoms, dtype=int)
        digitized_index = 0
        for i in range(n_atoms):
            if i in relevant_indices:
                all_layers_indices[i] = new_digitized[digitized_index]
                digitized_index += 1
        return list(all_layers_indices)

    @classmethod
    def get_layers_below_x(cls, ps: utils.PeriodicSystem, x: float, layers: List[int]):
        max_layer = int(np.max(layers))
        surface_z = np.mean([coord[2] for count, coord in enumerate(ps.atoms.positions) if layers[count] == max_layer])
        for i in range(max_layer, -1, -1):
            z = np.mean([coord[2] for count, coord in enumerate(ps.atoms.positions) if layers[count] == i])
            if surface_z - z > x:
                layers_to_constrain = i
                break
        else:
            raise RuntimeError("Could not determine layers below {:f}".format(x))
        return range(layers_to_constrain + 1)
