#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from copy import deepcopy
from dataclasses import dataclass, field
from functools import reduce
from math import radians, ceil, sqrt
from operator import mul
from random import randrange, shuffle, randint
from scipy.linalg import norm
from typing import List, Tuple, Dict, Union, Any, Optional, Set
import numpy as np
import warnings

# Third party imports
import scine_database as db
import scine_utilities as utils
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.analysis.adsorption import AdsorbateSiteFinder

# Local application imports
from scine_chemoton.filters.reactive_site_filters import ReactiveSiteFilter
from .inter_reactive_complexes import InterReactiveComplexes
from ..surfaces.periodic_utils import PeriodicUtils
from ..surfaces.pymatgen_interface import PmgInterface


class Adsorbate:
    """
    Holds some additional information about molecule useful for adsorption procedure
    """

    def __init__(self, structure: db.Structure, reactive_atoms: Union[List[int], None] = None,
                 complex_options: Optional[dict] = None) -> None:
        self._complex_options = complex_options
        self.atoms = structure.get_atoms()
        self.positions_2d = np.asarray([self.atoms.positions[:, 0], self.atoms.positions[:, 1]]).transpose()
        if reactive_atoms is not None:
            self.reactive_atoms = reactive_atoms
        else:
            self.reactive_atoms = list(range(len(self.atoms)))
        self.directions: Dict[Tuple[int], np.ndarray] = self._get_adsorbate_directions()
        self.vdw_values = np.asarray([utils.ElementInfo.vdw_radius(el) for el in self.atoms.elements])
        self.vdw_values_reactive = [v for i, v in enumerate(self.vdw_values) if i in self.reactive_atoms]
        self.vdw_avg = np.mean(self.vdw_values)
        self.vdw_avg_reactive = float(np.mean([v for i, v in enumerate(self.vdw_values) if i in self.reactive_atoms]))
        self.sites: Optional[
            List[int]] = None  # specifies the number of occupied symmetrically equivalent sites by one adsorbate

    def _get_adsorbate_directions(self) -> Dict[Tuple[int], np.ndarray]:
        """
        generates via the InterReactiveComplexes viable attack points for each atom of the adsorbate

        Returns
        -------
        directions : Dict[Tuple[int], np.ndarray]
            One np.ndarray of shape (n,3) per atom for which there is at least
            one attack point and index of that atom
        """
        inter_generator = InterReactiveComplexes()
        if self._complex_options is not None:
            inter_generator.set_options(self._complex_options)
        if len(self.atoms) == 1:
            return {
                (0,): np.array([
                    np.array([0.0, 0.0, utils.ElementInfo.vdw_radius(self.atoms.get_element(0))])
                ])
            }
        return inter_generator.get_attack_points_per_atom(self.atoms.positions, self.atoms.elements)

    def get_max_extension(self) -> float:
        """
        get the maximum distance between two atoms in the current conformation
        this can be used to estimate how much vacuum has to be added in the model to have a proper distance between
        images after adsorption

        Returns
        -------
        max_distance : float
            The maximum distance between two atoms
        """
        n_atoms = len(self.atoms)
        if n_atoms == 1:
            return 0.0
        n_pairs = int(round(n_atoms * (n_atoms - 1) / 2))
        distances = np.zeros(n_pairs)
        count = 0
        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):  # avoids double counting
                distances[count] = norm(self.atoms.get_position(i) - self.atoms.get_position(j))
                count += 1
        return float(np.max(distances))

    def atoms_from_positions(self, positions: np.ndarray) -> utils.AtomCollection:
        """
        get an AtomCollection from the positions of an unknown number of adsorbates

        Parameters
        ----------
        positions : np.ndarray of shape (n,3)
            The positions of the adsorbed molecules

        Returns
        -------
        ac : utils.AtomCollection
            atoms of the adsorbed molecules
        """
        n_adsorbed = (len(positions[:, 0]) / len(self.atoms))
        int_n_adsorbed = int(round(n_adsorbed))
        if abs(n_adsorbed - int_n_adsorbed) > 0.01:
            raise RuntimeError(f"Received positions {positions} are not an integer multiple of the"
                               f"adsorbate {len(self.atoms)} atoms")
        slab_ad_ele = self.atoms.elements * int_n_adsorbed
        return utils.AtomCollection(slab_ad_ele, positions)


@dataclass
class AdsorptionResult:
    original_ps: utils.PeriodicSystem
    original_slab_dict: Dict[str, Any]
    ps: utils.PeriodicSystem
    substrate_name: str
    site_name: str
    adsorbate: Adsorbate
    surface_close_atom_indices: List[int]
    reactive_index: int
    attack_direction_number: int
    coverage: float
    true_adsorption: bool = True
    count: Optional[int] = None
    adsorbing_atom_name: str = field(init=False)
    miller_index: str = field(init=False)
    pymatgen_structure: Structure = field(init=False)
    atoms: utils.AtomCollection = field(init=False)
    pbc: utils.PeriodicBoundaries = field(init=False)
    surface_atom_indices: Set[int] = field(init=False)
    slab_extension: List[int] = field(init=False)
    slab_dict: dict = field(init=False)

    def __post_init__(self):
        # shortcuts to ps members
        self.atoms = self.ps.atoms
        self.pbc = self.ps.pbc
        self.surface_atom_indices = self.ps.solid_state_atom_indices

        self.slab_extension = [int(round(self.ps.pbc.lengths[i] / self.original_ps.pbc.lengths[i]))
                               for i in range(3)]
        help_ps = self.original_ps * self.slab_extension
        if help_ps.pbc != self.ps.pbc:
            raise RuntimeError(f"The original PeriodicSystem and the current PeriodicSystem do not seem to fit"
                               f"together based on their PeriodicBoundaries {str(self.original_ps.pbc)} and "
                               f"{str(self.ps.pbc)}")

        # pymatgen objects
        self.pymatgen_structure = PmgInterface.to_mat_structure(self.ps)
        self.slab_dict = PmgInterface.update_slab_dict(self.original_slab_dict, self.ps)

        # various names
        self.adsorbing_atom_name: str = f"{self.adsorbate.atoms.get_element(self.reactive_index)}-{self.reactive_index}"
        if self.count is not None:
            self.site_name += f"-{self.count}"  # pylint: disable=no-member
        slab = Slab.from_dict(self.original_slab_dict)
        self.miller_index = ""
        for i in slab.miller_index:
            self.miller_index += str(i)


class NotEnoughSpaceInCellException(Exception):
    pass


class Adsorber:
    """
    Class that generates structures with molecules adsorbed on surfaces based on input coverage

    This class has three top level functions.
    Their use cases are:

        * cover_all_sites:
            The adsorbate only occupies one site each and each site shall be covered
        * generate_single_adsorbate_on_given_extension:
            The adsorbate might take up more than one site, but we only want to place a single adsorbate on a slab
            extended as directed by the input parameter
        * generate_adsorption_structures:
            The adsorbate might take up more than one site, we want rotamers or not, we only know the coverage that
            we want to achieve, but do not specify how many adsorbates on which size of the slab,
            this is determined via the coverage in the function.
            This function also recognizes, if the first use case would be correct and applies it
    """

    def __init__(self, slab_dict: dict, adsorption_sites: Dict[str, np.ndarray], wanted_coverage: float = 0.25,
                 adsorption_distance: Union[float, None] = None, maximum_extension: int = 20, eps: float = 0.01,
                 n_surface_atoms: int = 1, sites_are_bohr: bool = False) -> None:
        self.slab_dict = slab_dict
        self.adsorption_sites = adsorption_sites if sites_are_bohr \
            else PmgInterface.mat_sites_to_bohr(adsorption_sites)
        self.wanted_coverage = wanted_coverage
        self.adsorption_distance = adsorption_distance
        self.maximum_extension = maximum_extension
        self.eps = eps
        self.n_surface_atoms = n_surface_atoms

    def cover_all_sites(self, substrate_name: str, ps: utils.PeriodicSystem, adsorbate: Adsorbate,
                        rotamers: bool) -> List[AdsorptionResult]:
        """
        This function returns adsorbed structures, where all symmetrically equivalent sites are covered.
        This function does not work with an adsorbate, which covers more than one site

        Parameters
        ----------
        substrate_name : str
            The name of the substrate
        ps : utils.PeriodicSystem
            The slab where the adsorbate is placed on
        adsorbate : Adsorbate
            The adsorbate
        rotamers : bool
            Whether the single adsorbate shall be rotated randomly

        Returns
        -------
        results : List[AdsorptionResult]
            List of results; each result is a container class containing various meta-information and a single structure
        """
        highest_z = np.max(ps.atoms.positions[:, 2])
        results: List[AdsorptionResult] = []
        sites = self.adsorption_sites
        n_adsorbate_atoms = len(adsorbate.atoms)
        adsorption_max = max(adsorbate.vdw_values_reactive)
        adsorption_min = min(adsorbate.vdw_values_reactive)
        """
        this 4-fold for loop is present in all three top level functions with slight variations based on the
        information needed. We loop over:
            Different types of sites (top, hollow, bridge)
            Different sites present in the slab (e.g. fcc-hollow and hcp-hollow)
            The adsorbing atoms of the adsorbate
            The possible direction of adsorption for each reactive atom of the adsorbate
        Each loop adds another result (structure with additional information)
        """
        for key in sites:
            if key == "all" or len(sites[key]) == 0:
                continue
            site_coords = np.asarray(sites[key])
            # if multiple sites in one category, then include counter
            include_count = len(site_coords[:, 0]) > 1
            for count, site in enumerate(site_coords):
                site[2] = highest_z
                true_adsorption = self._true_adsorption_over_range(site, ps, adsorption_min, adsorption_max)
                scaled_pbc = ps.get_primitive_cell_system(1e-6, True).pbc
                sites_per_direction = [int(round(ps.pbc.lengths[i] / scaled_pbc.lengths[i])) for i in range(3)]
                for reactive in adsorbate.reactive_atoms:
                    if true_adsorption is None:
                        this_true_adsorption = PeriodicUtils.true_adsorption_site(
                            site, ps, adsorbate.vdw_values[reactive])
                    else:
                        this_true_adsorption = true_adsorption
                    if self.adsorption_distance is None:
                        closest_element, _ = PeriodicUtils.get_closest_element(ps.atoms, site)
                        self.adsorption_distance = adsorbate.vdw_values[reactive] + \
                            utils.ElementInfo.vdw_radius(closest_element)
                    for dircount, direction in enumerate(adsorbate.directions[(reactive,)]):
                        slab_ad_pos = self._position_adsorbate_atoms(site, adsorbate, ps, reactive, direction,
                                                                     rotamers=rotamers)
                        adsorbate_ps = utils.PeriodicSystem(scaled_pbc, adsorbate.atoms_from_positions(slab_ad_pos))
                        ps_shift = slab_ad_pos - adsorbate_ps.atoms.positions
                        adsorbate_ps *= sites_per_direction
                        adsorbate_pos = adsorbate_ps.atoms.positions
                        super_ps = deepcopy(ps)
                        if rotamers:
                            screening_site = site
                            screening_site[2] += adsorbate.vdw_avg_reactive
                            slab_direction = self._get_slab_direction(ps, screening_site)
                            for i in range(self.n_surface_atoms):
                                index = i * n_adsorbate_atoms
                                attempt = 0
                                while True:
                                    rotated_pos = PeriodicUtils.rotate_pos_around_vec(
                                        adsorbate_pos[index:index + n_adsorbate_atoms], slab_direction,
                                        origin=adsorbate_pos[index + reactive])
                                    rotated_atoms = adsorbate.atoms_from_positions(rotated_pos)
                                    for j, p in enumerate(rotated_atoms.positions):
                                        rotated_atoms.set_position(j, p + ps_shift[j])
                                    if not PeriodicUtils.periodic_clash(rotated_atoms, super_ps):
                                        super_ps.atoms += rotated_atoms
                                        break
                                    attempt += 1
                                    assert attempt < 10
                        else:
                            atoms = adsorbate_ps.atoms
                            for j, p in enumerate(atoms.positions):
                                atoms.set_position(j, p + ps_shift[j % n_adsorbate_atoms])
                            super_ps.atoms += atoms
                        count_arg = count if include_count else None
                        result = AdsorptionResult(
                            original_ps=ps,
                            original_slab_dict=self.slab_dict,
                            ps=super_ps,
                            substrate_name=substrate_name,
                            site_name=key,
                            adsorbate=adsorbate,
                            surface_close_atom_indices=PeriodicUtils.get_top_surface_atoms_indices(ps),
                            reactive_index=reactive,
                            attack_direction_number=dircount,
                            coverage=self.wanted_coverage,
                            true_adsorption=this_true_adsorption,
                            count=count_arg
                        )
                        results.append(result)
        return results

    def generate_single_adsorbate_on_given_extension(self, substrate_name: str, ps: utils.PeriodicSystem,
                                                     adsorbate: Adsorbate, extension: List[int],
                                                     rotamers: bool = False, check_size: bool = False) \
            -> List[AdsorptionResult]:
        """
        The generated structures consist of a slab formed by the input slab and an extension in x- and y- direction
        as specified via the extension input and a single adsorbate placed in the middle of the extended slab.

        Parameters
        ----------
        substrate_name : str
            The name of the substrate
        ps : utils.PeriodicSystem
            The slab where the adsorbate is placed on
        adsorbate : Adsorbate
            The adsorbate
        extension : List[int]
            The extension of the slab in x- and y-direction
        rotamers : bool
            Whether the single adsorbate shall be rotated randomly
        check_size : bool
            Whether the surface slab shall be checked to fit the maximally extended adsorbate. This option can lead to
            a bigger surface than given as an argument

        Returns
        -------
        results : List[AdsorptionResult]
            List of results; each result is a container class containing various meta-information and a single structure
        """
        if check_size:
            for i in range(2):
                min_necessary_extension = adsorbate.get_max_extension() / ps.pbc.lengths[i]
                if min_necessary_extension > extension[i]:
                    single_extension = int(ceil(min_necessary_extension))
                    extension[i] = single_extension
                    warnings.warn(f'The adsorbate might not fit on a slab with the given extension.'
                                  f'Increasing extension to {extension}')
        if sqrt(extension[0] * extension[1]) > self.maximum_extension:
            raise RuntimeError(f"Extension {extension} is over the set maximum extension of {self.maximum_extension}")
        super_ps = ps * np.array(extension)
        highest_z_slab_atoms = np.max([z for i, z in enumerate(super_ps.atoms.positions[:, 2])
                                       if i in super_ps.solid_state_atom_indices])
        highest_z = np.max(super_ps.atoms.positions[:, 2])
        adsorption_max = max(adsorbate.vdw_values_reactive)
        adsorption_min = min(adsorbate.vdw_values_reactive)
        results = []
        sites = dict(self.adsorption_sites)  # copy
        # for detail information about loops check cover_all_sites function
        for key in sites:
            if key == "all" or len(sites[key]) == 0:
                continue
            site_coords = np.asarray(sites[key])
            # include counter to avoid ambiguity with e.g. multiple hollow sites
            include_count = len(site_coords[:, 0]) > 1
            for count, site in enumerate(site_coords):
                # make sure the adsorbed molecule is above the slab
                site = super_ps.pbc.translate_positions_into_cell(site)[0]
                true_adsorption = self._true_adsorption_over_range(site, super_ps, adsorption_min, adsorption_max)
                for reactive in adsorbate.reactive_atoms:
                    if true_adsorption is None:
                        this_true_adsorption = PeriodicUtils.true_adsorption_site(
                            site, ps, adsorbate.vdw_values[reactive])
                    else:
                        this_true_adsorption = true_adsorption
                    site[2] = highest_z_slab_atoms if this_true_adsorption else highest_z
                    slab_indices = self._get_relevant_surface_indices_for_site(super_ps, key, site)
                    if self.adsorption_distance is None:
                        closest_element, _ = PeriodicUtils.get_closest_element(ps.atoms, site)
                        self.adsorption_distance = adsorbate.vdw_values[reactive] + \
                            utils.ElementInfo.vdw_radius(closest_element)
                    for dircount, direction in enumerate(adsorbate.directions[(reactive,)]):
                        slab_ad_pos = self._position_adsorbate_atoms(site, adsorbate, ps, reactive, direction,
                                                                     rotamers)
                        slab_ad_atoms = adsorbate.atoms_from_positions(slab_ad_pos)
                        if PeriodicUtils.periodic_clash(slab_ad_atoms, super_ps):
                            continue
                        adsorbed_super_ps = deepcopy(super_ps)
                        adsorbed_super_ps.atoms += slab_ad_atoms
                        # create result object and give to list
                        count_arg = count if include_count else None
                        result = AdsorptionResult(original_ps=ps,
                                                  original_slab_dict=self.slab_dict,
                                                  ps=adsorbed_super_ps,
                                                  substrate_name=substrate_name,
                                                  site_name=key,
                                                  adsorbate=adsorbate,
                                                  reactive_index=reactive,
                                                  surface_close_atom_indices=list(set(slab_indices)),
                                                  attack_direction_number=dircount,
                                                  coverage=self.wanted_coverage,
                                                  true_adsorption=this_true_adsorption,
                                                  count=count_arg
                                                  )
                        results.append(result)
        if len(results) == 0:
            raise NotEnoughSpaceInCellException('No placement was possible. The unit cell is probably too small for '
                                                'a second adsorbate.')
        return results

    def generate_adsorption_structures(self, substrate_name: str, ps: utils.PeriodicSystem, adsorbate: Adsorbate,
                                       rotamers: bool = True, multiple_molecules: bool = False) \
            -> List[AdsorptionResult]:
        """
        The generated structures have a flexible extension limited by the max_extension of the Adsorber class and a
        flexible number of adsorbed molecules.
        This function determines the smallest structure necessary to achieve the wanted coverage.
        This also includes bigger adsorbates, which occupy multiple sites at once. For bigger adsorbates,
        high percentage coverages may fail due to the existence of small gaps between randomly placed adsorbates,
        which cannot be filled

        Parameters
        ----------
        substrate_name : str
            The name of the substrate
        ps : utils.PeriodicSystem
            The slab where the adsorbate is placed on
        adsorbate : Adsorbate
            The adsorbate
        rotamers : bool
            whether different rotamers of the adsorbate shall be placed
        multiple_molecules : bool
            Whether the algorithm may place more than one adsorbate molecule

        Returns
        -------
        results : List[AdsorptionResult]
            List of results; each result is a container class containing various meta-information and a single structure
        """
        if adsorbate.sites is None:  # specifies the number of occupied symmetrically equivalent sites by one adsorbate
            adsorbate.sites = self._get_covered_sites_by_one_adsorbate(adsorbate, ps)
        big_adsorbate = any(site != 1 for site in adsorbate.sites)
        # easier creation, therefore own function
        if abs(self.wanted_coverage - 1) < self.eps and not big_adsorbate:
            if self.n_surface_atoms > 1 and not multiple_molecules:
                raise RuntimeError(f"Cannot achieve coverage of {self.wanted_coverage} with a single molecule")
            return self.cover_all_sites(substrate_name, ps, adsorbate, rotamers)
        # the slab is extended to achieve the right coverage for small adsorbates
        extension, needed_sites = self._determine_cell_extension_and_sites(allow_multiple_molecules=multiple_molecules)
        super_ps = ps * extension
        highest_z_slab_atoms = np.max([z for i, z in enumerate(super_ps.atoms.positions[:, 2])
                                       if i in super_ps.solid_state_atom_indices])
        highest_z = np.max(super_ps.atoms.positions[:, 2])
        adsorption_max = max(adsorbate.vdw_values_reactive)
        adsorption_min = min(adsorbate.vdw_values_reactive)

        results = []
        sites = self.adsorption_sites
        # keys correspond to types of sites (hollow, bridge, on-top)
        for key in sites:
            if key == "all" or len(sites[key]) == 0:
                continue
            site_coords = np.asarray(sites[key])
            # include counter to avoid ambiguity with e.g. multiple hollow sites
            include_count = len(site_coords[:, 0]) > 1
            for count, site in enumerate(site_coords):
                true_adsorption = self._true_adsorption_over_range(site, super_ps, adsorption_min, adsorption_max)
                # now generate random equivalent sites in right number for coverage
                if not big_adsorbate:
                    abs_sites = self._generate_sites_for_coverage(site, extension, needed_sites, ps.pbc)
                else:
                    abs_sites = np.array([])  # not used
                for reactive in adsorbate.reactive_atoms:
                    if true_adsorption is None:
                        this_true_adsorption = PeriodicUtils.true_adsorption_site(
                            site, ps, adsorbate.vdw_values[reactive])
                    else:
                        this_true_adsorption = true_adsorption
                    z_value = highest_z_slab_atoms if this_true_adsorption else highest_z
                    abs_sites[:, 2] = [z_value for _ in range(abs_sites.shape[0])]
                    site[2] = z_value
                    if self.adsorption_distance is None:
                        closest_element, _ = PeriodicUtils.get_closest_element(ps.atoms, site)
                        self.adsorption_distance = adsorbate.vdw_values[reactive] + \
                            utils.ElementInfo.vdw_radius(closest_element)
                    for dircount, direction in enumerate(adsorbate.directions[(reactive,)]):
                        current_super_ps = deepcopy(super_ps)
                        if not big_adsorbate:
                            reference_ps = super_ps
                            slab_ad_pos = self._position_adsorbate_atoms(abs_sites, adsorbate, ps, reactive,
                                                                         direction, rotamers)
                            current_super_ps.atoms += adsorbate.atoms_from_positions(slab_ad_pos)
                        else:
                            # if big adsorbate, the needed coverage can vary with direction and atom
                            # --> reevaluate extension and sites for each again, not necessary for small adsorbate
                            current_super_ps, extension = self._reevaluate_sites(site, ps, adsorbate, rotamers,
                                                                                 reactive, direction,
                                                                                 multiple_molecules)
                            reference_ps = super_ps * extension
                        # need to get relevant surface indices
                        # first determine all adsorbate reactive positions
                        n_atoms = len(current_super_ps.atoms)
                        n_atoms_before_adsorption = len(reference_ps.atoms)
                        n_adsorbate_atoms = n_atoms - n_atoms_before_adsorption
                        n_atoms_per_adsorbate = len(adsorbate.atoms)
                        if n_adsorbate_atoms % n_atoms_per_adsorbate != 0:
                            raise RuntimeError("Could not determine the relevant surface indices")
                        all_indices = []
                        # assume all adsorbing atoms are at the back of the atoms list
                        for i in range(n_atoms_before_adsorption, n_atoms, n_atoms_per_adsorbate):
                            this_adsorbate_pos = current_super_ps.atoms[i:i + n_atoms_per_adsorbate].positions
                            relevant_site = this_adsorbate_pos[reactive]
                            all_indices += self._get_relevant_surface_indices_for_site(current_super_ps,
                                                                                       key, relevant_site)
                        # create result object and give to list
                        count_arg = count if include_count else None
                        result = AdsorptionResult(original_ps=ps,
                                                  original_slab_dict=self.slab_dict,
                                                  ps=current_super_ps,
                                                  substrate_name=substrate_name,
                                                  site_name=key,
                                                  adsorbate=adsorbate,
                                                  surface_close_atom_indices=list(set(all_indices)),
                                                  reactive_index=reactive,
                                                  attack_direction_number=dircount,
                                                  coverage=self.wanted_coverage,
                                                  true_adsorption=this_true_adsorption,
                                                  count=count_arg
                                                  )
                        results.append(result)
        return results

    @staticmethod
    def _true_adsorption_over_range(site: np.ndarray, ps: utils.PeriodicSystem,
                                    min_adsorbate_vdw: float, max_adsorbate_vdw: float) -> Optional[bool]:
        true_adsorption_min = PeriodicUtils.true_adsorption_site(site, ps, min_adsorbate_vdw)
        true_adsorption_max = PeriodicUtils.true_adsorption_site(site, ps, max_adsorbate_vdw)
        true_adsorption_is_clear: bool = true_adsorption_min == true_adsorption_max
        if true_adsorption_is_clear:
            return true_adsorption_min
        return None

    @staticmethod
    def _get_relevant_surface_indices_for_site(ps: utils.PeriodicSystem, site_name: str, site: np.ndarray) -> List[int]:
        result: List[int] = []
        # determine required number of indices
        n_sites_for_name = {"ontop": 1, "bridge": 2, "hollow": 3}
        try:
            n_needed = n_sites_for_name[site_name]
        except KeyError as e:
            raise RuntimeError(f"Number of surface indices for site {site_name} is unknown.") from e
        # only take solid state indices to avoid confusion with existing adsorbate
        atoms = utils.AtomCollection(len(ps.atoms) - len(ps.solid_state_atom_indices))
        for i, atom in enumerate(ps.atoms):
            if i in ps.solid_state_atom_indices:
                atoms.push_back(atom)
        # get indices
        for _ in range(n_needed):
            _, index = PeriodicUtils.get_closest_element(atoms, site)
            result.append(index)
            del atoms[index]
        return result

    def _determine_cell_extension_and_sites(self, allow_multiple_molecules: bool = True, n_covered_sites: int = 1) \
            -> Tuple[List[int], int]:
        """
        Calculate the necessary number of sites on the smallest possible extension of the slab to achieve the wanted
        coverage

        Parameters
        ----------
        allow_multiple_molecules : bool
            Whether the algorithm may place more than one adsorbate molecule
        n_covered_sites : int
            The number of sites that should be covered

        Returns
        -------
        extension, n_sites : Tuple[int, int]
            the necessary extension and number of sites
        """
        for i in range(self.maximum_extension):
            extension = i + 1
            all_sites = self.n_surface_atoms * extension ** 2
            possible_deltas = np.zeros(all_sites + 1)
            if allow_multiple_molecules:
                for j in range(all_sites + 1):
                    coverage = j / all_sites
                    possible_deltas[j] = abs(coverage - self.wanted_coverage)
                if np.min(possible_deltas) < self.eps:
                    return [extension, extension, 1], int(np.argmin(possible_deltas))
            else:
                coverage = n_covered_sites / all_sites
                if abs(coverage - self.wanted_coverage) < self.eps:
                    return [extension, extension, 1], 1
        raise RuntimeError("Could not get right coverage within allowed extension of surface.")

    def _generate_sites_for_coverage(self, site: np.ndarray, extension: List[int], needed_sites: int,
                                     pbc: utils.PeriodicBoundaries) -> np.ndarray:
        """
        generate random but symmetrically equivalent sites on extended slab

        Parameters
        ----------
        site : np.ndarray of len 3
            The coordinates of one site
        extension : List[int]
            The extension of the slab in x-, y, and z-direction
        needed_sites : int
            The number of sites to generate
        pbc : utils.PeriodicBoundaries
            The periodic boundaries of the slab

        Returns
        -------
        result_sites : np.ndarray of shape (needed_sites, 3)
            randomly generated but symmetrically equivalent sites
        """
        within_site = pbc.translate_positions_into_cell(site)[0]
        result_sites = np.zeros(shape=(needed_sites, 3))
        occupied_sites: List[Tuple[int, int]] = []
        for i in range(needed_sites):
            result_sites[i] = self._get_unoccupied_site(pbc, within_site, occupied_sites, extension,
                                                        allowed_attempts=int(1e12))
            if len(result_sites) == 0:
                raise RuntimeError("Could not find any left unoccupied site")
        return result_sites

    def _position_adsorbate_atoms(self, sites: np.ndarray, adsorbate: Adsorbate,
                                  ps: utils.PeriodicSystem,
                                  reactive_atom: int, direction: np.ndarray, rotamers: bool,
                                  angle: Optional[float] = None,
                                  max_attempts: int = 100) -> np.ndarray:
        """
        position adsorbate on given sites

        Parameters
        ----------
        sites : np.ndarray of shape (n, 3) or len 3
            The coordinates of the sites to place the adsorbate above
        adsorbate : Adsorbate
            The adsorbate
        ps : utils.PeriodicSystem
            The slab where the adsorbate is placed on
        reactive_atom : int
            The index of the adsorbing atom of the adsorbate
        direction : np.ndarray
            The relative vector to adsorb coming from the adsorbing atom
        rotamers : bool
            whether different rotamers of the adsorbate shall be placed
        angle : Union[float, None]
            The angle the rotamers shall be rotated; if None the angle is random
        max_attempts : int
            The maximum number of attempts to find a non-clashing position

        Returns
        -------
        all_ad_positions : np.ndarray of shape (n, 3)
            The positions of all placed adsorbates
        """
        # the enumerated sites here correspond to sites of identical type (e.g. all on-top)
        # they are only shifted relative for right coverage or there is only 1 site
        n_single_ad_atoms = len(adsorbate.atoms)
        if sites.ndim == 2:  # sites is an np.ndarray -> multiple sites
            n_sites = len(sites[:, 0])
        else:  # only 1 site present, put it in list, so code below works like with the np.ndarray
            n_sites = 1
            sites = np.array([sites])
        # determine attack vector for slab in some screening distance to avoid vdw overlap during determination
        screening_site = sites[0].copy()
        screening_site[2] += adsorbate.vdw_avg_reactive
        # to avoid unnecessary calculations use first site for slab direction
        slab_direction = self._get_slab_direction(ps, screening_site)
        all_ad_positions: Optional[np.ndarray] = None
        while all_ad_positions is None:
            reference_ps = deepcopy(ps)
            all_ad_positions = np.zeros(shape=(n_single_ad_atoms * n_sites, 3))  # result of this function
            for count, site in enumerate(sites):
                # if we rotate things and have multiple sites that we are covering at once, we may clash
                clash_is_avoidable: bool = rotamers and angle is None  # if we change the angle and the angle is random
                attempts = 0
                while True:
                    # shift attack point to site
                    site_ad_pos = adsorbate.atoms.positions.copy()
                    site_ad_pos = PeriodicUtils.translate_position_from_point_to_point(
                        site_ad_pos, direction, site)
                    # attack vector of adsorbate
                    rel_direction = site - site_ad_pos[reactive_atom]
                    # rotate adsorbate so its vector is aligned with the slab vector
                    rotM = InterReactiveComplexes.rotation_to_vector(rel_direction, -slab_direction)
                    site_ad_pos = (rotM.T.dot(site_ad_pos.T)).T
                    if rotamers:
                        # rotate molecule with random or given angle
                        site_ad_pos = PeriodicUtils.rotate_pos_around_vec(site_ad_pos, slab_direction,
                                                                          origin=site_ad_pos[reactive_atom],
                                                                          angle=angle)
                    site_ad_pos = PeriodicUtils.translate_position_from_point_to_point(
                        site_ad_pos, site_ad_pos[reactive_atom], site)
                    # reactive atom now on site --> too close --> move away along the slab_direction vector
                    # (default 2 vdw averages of adsorbate)
                    distance = self.adsorption_distance
                    trans_vec = distance * slab_direction / norm(slab_direction)
                    site_ad_pos = np.add(trans_vec, site_ad_pos)
                    site_ad_atoms = adsorbate.atoms_from_positions(site_ad_pos)
                    if not clash_is_avoidable or not PeriodicUtils.periodic_clash(site_ad_atoms, reference_ps):
                        break
                    attempts += 1
                    if attempts > max_attempts:
                        all_ad_positions = None
                        break
                if all_ad_positions is None:
                    break
                reference_ps.atoms += site_ad_atoms
                # write finished position into matrix
                index = count * n_single_ad_atoms
                all_ad_positions[index: index + n_single_ad_atoms] = site_ad_pos
        return all_ad_positions

    def _reevaluate_sites(self, site: np.ndarray, ps: utils.PeriodicSystem, adsorbate: Adsorbate, rotamers: bool,
                          reactive: int, direction: np.ndarray, allow_multiple_molecules: bool) \
            -> Tuple[utils.PeriodicSystem, List[int]]:
        """
        for bigger adsorbate, this function is needed to determine the sites due to some additional necessary logic

        Parameters
        ----------
        site : np.ndarray of len 3
            The coordinates of the site
        ps : utils.PeriodicSystem
            The slab where the adsorbate is placed on
        adsorbate : Adsorbate
            The adsorbate
        rotamers : bool
            whether different rotamers of the adsorbate shall be placed
        reactive : int
            The index of the adsorbing atom of the adsorbate
        direction : np.ndarray
            The relative vector to adsorb coming from the adsorbing atom
        allow_multiple_molecules : bool
            Whether the algorithm may place more than one adsorbate molecule

        Returns
        -------
        superslab_ad_atoms, extension : Tuple[utils.PeriodicSystem, List[int]]
            The AtomCollection of the placed adsorbate and the slab and how much the slab was extended in the x- and
            y-direction and z-direction
        """
        if rotamers:  # number of covered sites different for rotamers -> get distribution depending on angle
            distribution = self._get_distribution_of_coverages(site, ps, adsorbate, reactive, direction)
        else:
            ad_pos = self._position_adsorbate_atoms(site, adsorbate, ps, reactive, direction, False)
            covered = len(self._get_covered_sites_by_positions(site, adsorbate, ps.pbc, ad_pos))
            distribution = np.array([np.array([covered, 1.0, [0.0]])])
        extension = [0, 0, 0]
        atoms = None
        while atoms is None:
            if allow_multiple_molecules:
                selection, extension = self._get_big_states_and_extension(distribution, extension)
                needed_sites = sum(selection)
                super_ps = ps * extension
                atoms = self._generate_atoms_for_coverage_big_molecules(site, extension, needed_sites, ps, adsorbate,
                                                                        reactive, direction, distribution, selection)
            else:
                if rotamers:
                    d = np.asarray(distribution)
                    random_state_number = randrange(0, len(d[:, 0]))
                    n_covered_sites = d[random_state_number][0]
                    angles = d[random_state_number][2]
                    angle = angles[randrange(0, len(angles))]
                else:
                    n_covered_sites = covered
                    angle = None
                extension, needed_sites = self._determine_cell_extension_and_sites(
                    allow_multiple_molecules=allow_multiple_molecules,
                    n_covered_sites=n_covered_sites)
                super_ps = ps * extension
                slab_ad_pos = self._position_adsorbate_atoms(site, adsorbate, ps, reactive,
                                                             direction, rotamers, angle=angle)
                atoms = adsorbate.atoms_from_positions(slab_ad_pos)

        # final structure
        for a in atoms:
            super_ps.atoms.push_back(a)
        return super_ps, extension

    def _get_distribution_of_coverages(self, site: np.ndarray, ps: utils.PeriodicSystem, adsorbate: Adsorbate,
                                       reactive: int, direction: np.ndarray) -> np.ndarray:
        """
        determine distribution of covered sites depending on angle of rotation by sampling all angles and counting
        covered sites

        Parameters
        ----------
        site : np.ndarray of len 3
            The coordinates of the site
        ps : utils.PeriodicSystem
            The slab where the adsorbate is placed on
        adsorbate : Adsorbate
            The adsorbate
        reactive : int
            The index of the adsorbing atom of the adsorbate
        direction : np.ndarray
            The relative vector to adsorb coming from the adsorbing atom

        Returns
        -------
        dist_list : np.ndarray of shape (36, 3)
            The number of covered sites, the percentage of all covered states that cover that many sites
            and the list of angles that lead to this cover in each row
        """
        positions = self._position_adsorbate_atoms(site, adsorbate, ps, reactive, direction, False)
        vec = positions[reactive] - site
        interval = 10
        n_rot = int(round(360 / interval))
        dist_dict: Dict[int, float] = {}
        angles_dict: Dict[int, List[float]] = {}
        for i in range(0, 360, interval):
            angle = radians(i)
            tmp_pos = PeriodicUtils.rotate_pos_around_vec(positions, vec, origin=positions[reactive], angle=angle)
            n_covered = len(self._get_covered_sites_by_positions(site, adsorbate, ps.pbc, tmp_pos))
            if str(n_covered) in dist_dict:  # add to existing entry
                dist_dict[n_covered] += 1 / n_rot
                angles_dict[n_covered].append(angle)
            else:  # make new entry
                dist_dict[n_covered] = 1 / n_rot
                angles_dict[n_covered] = [angle]
        dist_list = []
        for key in dist_dict:
            dist_list.append((key, dist_dict[key], angles_dict[key]))
        return np.asarray(dist_list)

    def _get_big_states_and_extension(self, distribution: np.ndarray, start_extension: List[int]) \
            -> Tuple[List[int], List[int]]:
        """
        This function could be used iteratively in the algorithm.
        It tries to find a number of placed adsorbates in different rotations at an extended slab that achieves both
        the wanted coverage and is also fairly similar in its distribution of covered states to the one already
        observed in a scan of all angles.
        This should be achieved with the smallest possible extension. Because the placement of big adsorbates might
        still fail later due to the random placement and the possibility that small uncovered areas between the
        adsorbates can never be covered, the function accepts a start value for the extension from which on onwards
        it tries to find the smallest possible extension until it reaches the maximum allowed extension and raises an
        exception

        Parameters
        ----------
        distribution : np.ndarray of shape (36, 3)
            The number of covered sites, the percentage of all covered states that cover that many sites
            and the list of angles that lead to this cover in each row
        start_extension : List[int]
            The minimum extension the algorithm is started with

        Returns
        -------
        selection, extension : Tuple[List[int], int]
            The number to select from each existing state in the distribution in the order as they are appearing in
            the distribution and the extension needed for this selection
        """
        single_covers = np.asarray(distribution[:, 0])
        population = np.asarray(distribution[:, 1])
        min_covered = np.min(single_covers)
        max_covered = np.max(single_covers)
        extension = deepcopy(start_extension)
        while extension[0] * extension[1] < self.maximum_extension:
            index = 0 if extension[0] < extension[1] else 1
            extension[index] += 1
            if min_covered > extension[0] * extension[1]:
                # one adsorbate with the lowest coverage already bigger than slab
                continue
            # numbers to place the biggest coverage state and lowest to get right coverage within margin
            max_n = int(round((self.wanted_coverage + self.eps) * extension[0] * extension[1] / min_covered))
            min_n = int(round((self.wanted_coverage - self.eps) * extension[0] * extension[1] / max_covered))
            if min_n == 0:
                min_n = 1
            dist_threshold = 0.1
            for n in range(min_n, max_n):
                # get range for each state that is still roughly within wanted distribution
                ranges: List[Tuple[int, int]] = []
                for p in population:
                    max_int = int((p + dist_threshold) * n)
                    min_int = int((p - dist_threshold) * n)
                    if min_int == 0:
                        min_int = 1
                    ranges.append((min_int, max_int + 1))
                # generate all possible combinations of different ranges
                operations = reduce(mul, (r[1] - r[0] for r in ranges)) - 1
                selection = [r[0] for r in ranges]
                # generated combinations are ascending
                pos = len(ranges) - 1
                increments = 0
                # first selection has indiv. check, because not encountered in while loop
                # if placements bigger than max_n, coverage not possible anyway
                if sum(selection) > max_n:
                    break
                if self._finish_check(single_covers, selection, extension, population, dist_threshold, n):
                    return selection, extension
                while increments < operations:
                    if selection[pos] == ranges[pos][1] - 1:
                        selection[pos] = ranges[pos][0]
                        pos -= 1
                    else:
                        selection[pos] += 1
                        increments += 1
                        pos = len(ranges) - 1  # increment the innermost loop
                        if sum(selection) > max_n:
                            break
                        if self._finish_check(single_covers, selection, extension, population, dist_threshold, n):
                            return selection, extension
        raise RuntimeError("Could not get right coverage within allowed extension of surface and error margin of "
                           "coverage.")

    def _finish_check(self, single_covers: np.ndarray, selection: List[int], extension: List[int],
                      population: np.ndarray, threshold: float, n: int) -> bool:
        return self._converged_coverage(single_covers, selection, extension) \
            and self._within_distribution(selection, population, threshold, n)

    def _converged_coverage(self, single_covers: np.ndarray, selection: List[int], extension: List[int]) -> bool:
        covered_sites = self._n_covered_sites(single_covers, selection)
        coverage = covered_sites / (extension[0] * extension[1])
        return abs(coverage - self.wanted_coverage) < self.eps

    @staticmethod
    def _within_distribution(selection: List[int], population: np.ndarray, threshold: float, n: int) -> bool:
        return all(abs(s / n - p) < threshold for s, p in zip(selection, population))

    @staticmethod
    def _n_covered_sites(single_covers: np.ndarray, selection: List[int]) -> int:
        return int(np.dot(np.asarray(single_covers), np.asarray(selection)))

    def _generate_atoms_for_coverage_big_molecules(self, site: np.ndarray, extension: List[int], needed_sites: int,
                                                   ps: utils.PeriodicSystem, adsorbate: Adsorbate, reactive: int,
                                                   direction: np.ndarray, distribution: np.ndarray,
                                                   selection: List[int]) -> Union[None, utils.AtomCollection]:
        """
        Place adsorbate atoms which cover more than 1 site per molecule on the slab and return their atoms
        If the placement fails because of a too high wanted coverage, this function returns and empty list

        Parameters
        ----------
        site : np.ndarray of len 3
            The coordinates of the site
        extension : List[int]
            The extension of the slab in the x- and y-direction
        needed_sites : int
            The needed number of placed adsorbates
        ps : utils.PeriodicSystem
            The slab where the adsorbate is placed on
        adsorbate : Adsorbate
            The adsorbate
        reactive : int
            The index of the adsorbing atom of the adsorbate
        direction : np.ndarray
            The relative vector to adsorb coming from the adsorbing atom
        distribution : np.ndarray of shape (36, 3)
            The number of covered sites, the percentage of all covered states that cover that many sites
            and the list of angles that lead to this cover in each row
        selection : List[int]
            The number to select from each state of the adsorbate

        Returns
        -------
        superslab_ad_atoms : utils.AtomCollection
            The AtomCollection of the placed adsorbates
        """
        # read data from distribution and shuffle angles for random selection
        adsorbed: List[np.ndarray] = []
        # single_covers = np.asarray(distribution[:, 0])  # for debug below
        angles = np.asarray(distribution[:, 2])
        for a in angles:
            shuffle(a)
        # ensure site is in bottom left cell, so the random translation stays on superslab
        site = ps.pbc.translate_positions_into_cell(site)[0]
        # used later for detection of adsorbate clashing with image of other adsorbates
        super_ps = ps * extension
        occupied_sites: List[Tuple[int, int]] = []
        n_sites = 0
        # counter for the states in the distribution, with +1 offset due to summation over slice
        place_state = 1
        counter = 0
        attempts = 0
        failed = []
        while counter < needed_sites:
            attempts += 1
            if counter == sum(selection[:place_state]):
                place_state += 1
            new_site = self._get_unoccupied_site(ps.pbc, site, occupied_sites, extension)
            if attempts > extension[0] * extension[1] or len(new_site) == 0:
                # failed to achieve coverage
                """ DEBUG
                from pymatgen.io.cif import CifWriter
                n_atoms = len(adsorbate.atoms)
                all_ad_pos = np.zeros(shape=(len(adsorbed) * n_atoms, 3))
                for i in range(len(adsorbed)):
                    index = i * n_atoms
                    all_ad_pos[index : index + n_atoms] = adsorbed[i]
                ad = adsorbate.atoms_from_positions(all_ad_pos)
                superslab_atoms = PeriodicUtils.get_cart_atoms_from_slab(superslab)
                for a in ad:
                    super_ps.atoms.push_back(a)
                structure = PmgInterface.to_mat_structure(super_ps, super_ps.atoms)
                writer = CifWriter(structure)
                writer.write_file("failed.cif")
                failed_atoms = []
                for f in failed:
                    failed_atoms.append(adsorbate.atoms_from_positions(f))
                for i, fs in enumerate(failed_atoms):
                    utils.IO.write("failed-"+str(i)+".xyz", fs)
                """
                # ensures retry with bigger extension
                return None
            candidate = self._position_adsorbate_atoms(new_site, adsorbate, ps, reactive, direction, False)
            # vector to rotate around
            adsorbed_vec = candidate[reactive] - new_site
            # possible angles to achieve the wanted state to get right distribution of states
            avail_angles = angles[place_state - 1]
            # checks for clashes, allows one rotation, appends non-clashing to list 'adsorbed'
            found_new = self._check_clashes(candidate, adsorbate, adsorbed_vec, reactive, adsorbed, super_ps.pbc,
                                            avail_angles)
            if found_new:
                attempts = 0
                counter += 1
                # to limit unnecessary computations,
                # the covered sites by this new adsorbate are also labeled as occupied
                covered = self._get_covered_sites_by_positions(new_site, adsorbate, ps.pbc, adsorbed[-1])
                n_sites += len(covered)
                rel_site = ps.pbc.transform(new_site, False)[0]
                for c in covered:
                    x_occ = c[0] - rel_site[0]
                    y_occ = c[1] - rel_site[1]
                    if x_occ < 0:
                        x_occ += extension[0]
                    if y_occ < 0:
                        y_occ += extension[1]
                    occupied_sites.append((int(round(x_occ)), int(round(y_occ))))
            else:
                failed.append(candidate)

        # achieved coverage
        n_atoms = len(adsorbate.atoms)
        all_ad_pos = np.zeros(shape=(len(adsorbed) * n_atoms, 3))
        for i in range(len(adsorbed)):
            index = i * n_atoms
            all_ad_pos[index: index + n_atoms] = adsorbed[i]
        return adsorbate.atoms_from_positions(all_ad_pos)

    @staticmethod
    def _get_unoccupied_site(pbc: utils.PeriodicBoundaries, site: np.ndarray,
                             occupied_sites: List[Tuple[int, int]], extension: List[int],
                             allowed_attempts: int = 1000) -> np.ndarray:
        """
        find a site on the superslab which is not yet occupied

        Parameters
        ----------
        slab : Slab
            The slab where the adsorbate is placed on
        site : np.ndarray of len 3
            The coordinates of the site
        occupied_sites : List[Tuple[int, int]]
            List of the sites in 2d cell coordinates relative to the site, which are already occupied
        extension : List[int]
            The extension of the slab in the x- and y-direction
        allowed_attempts : int
            The maximum number of tries to find an occupied site

        Returns
        -------
        unoccupied_site : np.ndarray
            The Cartesian coordinates of the unoccupied site
        """
        rel_site = pbc.transform(site, False)[0]
        found_new = False
        attempts = 0
        while not found_new:
            attempts += 1
            if attempts > allowed_attempts:
                # Was not able to achieve wanted coverage, trying again
                return np.array([])
            rand_x = randint(0, extension[0] - 1)
            rand_y = randint(0, extension[1] - 1)
            rand = (rand_x, rand_y)
            if all(rand != occ for occ in occupied_sites):
                occupied_sites.append(rand)
                # position adsorbate on possible position with angle=0
                rel_site += np.array([rand_x, rand_y, 0.0])
                return pbc.transform(rel_site)[0]
        return np.array([])

    @staticmethod
    def _check_clashes(candidate: np.ndarray, adsorbate: Adsorbate, adsorbed_vec: np.ndarray,
                       reactive: int, adsorbed: List[np.ndarray], pbc: utils.PeriodicBoundaries,
                       avail_angles: Union[List[float], None] = None) -> bool:
        """
        Check if the candidate positions of the adsorbate clash with any other position of the already adsorbed
        molecules.
        The nearest image convention is applied

        Parameters
        ----------
        candidate : np.ndarray of shape (n, 3)
            The positions of the possible new adsorbate
        adsorbate : Adsorbate
            The adsorbate
        adsorbed_vec : np.ndarray of len 3
            The vector between the adsorbing atom of the candidate and the site it is adsorbed on
        reactive : int
            The index of the adsorbing atom of the adsorbate
        adsorbed : List[np.ndarray of shape (n, 3)]
            A list of PositionCollections of already adsorbed molecules. The candidate is added to this list,
            if no clash is found
        pbc : utils.PeriodicBoundaries
            The periodic boundary conditions
        avail_angles : List[float]
            The angles that the candidate can be rotated to still cover the same amount of sites

        Returns
        -------
        not_clashing : bool
            if the candidate is valid (True) or if it clashes with previous adsorbed molecules (False)
        """
        if avail_angles is None:
            avail_angles = [0.0]
        for angle in avail_angles:
            candidate = PeriodicUtils.rotate_pos_around_vec(candidate, adsorbed_vec, origin=candidate[reactive],
                                                            angle=angle)
            if all(not PeriodicUtils.clashing_adsorbates(positions, candidate, adsorbate.vdw_values, pbc)
                   for positions in adsorbed):
                adsorbed.append(candidate)
                return True
            # revert previous rotation
            candidate = PeriodicUtils.rotate_pos_around_vec(candidate, adsorbed_vec, origin=candidate[reactive],
                                                            angle=-angle)
        return False

    def _get_covered_sites_by_one_adsorbate(self, adsorbate: Adsorbate, ps: utils.PeriodicSystem) -> List[int]:
        """
        This sets the covered symmetrically equivalent sites for the given conformer of the adsorbate for each
        reactive atom and each direction of adsorption. The result is written into the Adsorbate class variable
        sites as a List of int.

        Parameters
        ----------
        adsorbate : Adsorbate
            The adsorbate
        ps : utils.PeriodicSystem
            The slab where the adsorbate is placed on
        """
        old_distance_value = self.adsorption_distance
        if self.adsorption_distance is None:
            self.adsorption_distance = 2 * adsorbate.vdw_avg_reactive
        adsorbate_sites = []
        # get first site, as they are all equally spaced, ontop should always be present
        site = np.asarray(self.adsorption_sites["ontop"])[0]
        # ensure it is in unit cell
        site = ps.pbc.translate_positions_into_cell(site)[0]
        # adsorbate's range can vary depending on orientation --> try all possible orientations
        for reactive in adsorbate.reactive_atoms:
            for direction in adsorbate.directions[(reactive,)]:
                ad_pos = self._position_adsorbate_atoms(site, adsorbate, ps, reactive, direction, rotamers=False)
                adsorbate_sites.append(len(self._get_covered_sites_by_positions(site, adsorbate, ps.pbc, ad_pos)))
        self.adsorption_distance = old_distance_value
        return adsorbate_sites

    def _get_covered_sites_by_positions(self, site: np.ndarray, adsorbate: Adsorbate, pbc: utils.PeriodicBoundaries,
                                        positions: np.ndarray) -> np.ndarray:
        """
        Get all symmetrically equivalent sites that are covered by the adsorbate in the specified position

        Parameters
        ----------
        site : np.ndarray
            The site where the adsorbate is adsorbed to and determines the symmetrically equivalent sites
        adsorbate : Adsorbate
            The adsorbate
        pbc : utils.PeriodicBoundaries
            The periodic boundary conditions
        positions : np.ndarray of shape (n, 3)
            The positions of the adsorbate's atoms

        Returns
        -------
        sites : np.ndarray of shape (n,3)
            A matrix of cell coordinates relative to the input site. It contains the n sites that are covered by the
            adsorbate.
        """
        # transform both to cell coordinates
        rel_site = pbc.transform(site, False)[0]
        rel_pos = pbc.transform(positions, False)
        # get rectangle covered by adsorbate
        corners = self._rectangle_corners_of_positions(rel_pos, adsorbate, pbc)
        # difference in rel. coordinates between symmetrically equivalent sites in one dimension
        step = 1.0 / sqrt(self.n_surface_atoms)
        # set up grid
        ranges = self._get_range_of_grid(corners, rel_site, step)
        # fill grid with directly above adsorbate
        grid = self._grid_with_direct_hits(positions, adsorbate, pbc, ranges)
        # fill in all grid values as hit that are surrounded by hit values
        xlimit = len(grid)
        ylimit = len(grid[0])
        # cycle all voxels
        for x in range(xlimit):
            for y in range(ylimit):
                if grid[x][y] == 0:
                    # get all visits by fill4 algorithm
                    visit_list = self._fill4(x, y, xlimit, ylimit, grid)
                    # ensure all visits are set to 1
                    for coord in visit_list:
                        grid[coord[0]][coord[1]] = 1
        # translate grid entries to coordinates
        sites = []
        i = ranges[0][0] - step
        for x in range(xlimit):
            i += step
            j = ranges[1][0] - step
            for y in range(ylimit):
                j += step
                if grid[x][y] == 1:
                    sites.append(np.array([i, j, rel_site[2]]))
        return np.asarray(sites)

    @staticmethod
    def _rectangle_corners_of_positions(positions: np.ndarray, adsorbate: Adsorbate,
                                        pbc: utils.PeriodicBoundaries) \
            -> Tuple[float, float, float, float]:
        """
        Draw a rectangle around the x- and y- coordinates + vdw spheres of the input adsorbate positions and return the
        coordinates of the corners

        Parameters
        ----------
        positions : np.ndarray of shape (n, 3)
            The positions of the adsorbate's atoms
        adsorbate : Adsorbate
            The adsorbate
        pbc : PeriodicBoundaries
            The periodic boundary conditions

        Returns
        -------
        corners : Tuple[float, float, float, float]
            The coordinates of the corners of a rectangle as smallest x, smallest y, biggest x, and biggest y
        """
        a = positions
        x_min = np.min(a[:, 0]) - (adsorbate.vdw_values[np.argmin(a[:, 0])] / pbc.lengths[0])
        y_min = np.min(a[:, 1]) - (adsorbate.vdw_values[np.argmin(a[:, 1])] / pbc.lengths[1])
        x_max = np.max(a[:, 0]) + (adsorbate.vdw_values[np.argmax(a[:, 0])] / pbc.lengths[0])
        y_max = np.max(a[:, 1]) + (adsorbate.vdw_values[np.argmax(a[:, 1])] / pbc.lengths[1])
        return x_min, y_min, x_max, y_max

    @staticmethod
    def _get_range_of_grid(corners: Tuple[float, float, float, float], site: np.ndarray, step: float) -> Tuple[
            np.ndarray, np.ndarray]:
        """
        Get the covered coordinates of the grid in x- and y-direction from the corners of the 2d rectangle,
        the site that is adsorbed to, and the stepsize to take from the site in each direction

        Parameters
        ----------
        corners : Tuple[float, float, float, float]
            The coordinates of the corners of a rectangle as smallest x, smallest y, biggest x, and biggest y
        site : np.ndarray of len 3
            The site that is covered by adsorbate
        step : float
            The step size of the constructed grid from the original site in each direction

        Returns
        -------
        ranges : Tuple[np.ndarray, np.ndarray]
            The positions of the grid in x- and y-direction in cell coordinates.
        """
        xmin = site[0]
        xmax = site[0]
        ymin = site[1]
        ymax = site[1]
        while xmin > corners[0]:
            xmin -= step
        while xmax < corners[2]:
            xmax += step
        while ymin > corners[1]:
            ymin -= step
        while ymax < corners[3]:
            ymax += step
        # give an additional step for min and max
        # reason is SAS-like check if avg. vdw sphere of adsorbate would fit on grid sites --> extra space to the side
        x_range = np.arange(xmin - step, 2 * step + xmax, step)
        y_range = np.arange(ymin - step, 2 * step + ymax, step)
        return x_range, y_range

    @staticmethod
    def _grid_with_direct_hits(positions: np.ndarray, adsorbate: Adsorbate, pbc: utils.PeriodicBoundaries,
                               ranges: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Construct the two-dimensional grid that is filled with 0 and 1. 1 corresponds to a position,
        where the vdw sphere of an atom of the adsorbate is directly above.

        Parameters
        ----------
        positions : np.ndarray of shape (n, 3)
            The positions of the adsorbate's atoms
        adsorbate : Adsorbate
            The adsorbate
        pbc : utils.PeriodicBoundaries
            The periodic boundary conditions
        ranges : Tuple[np.ndarray, np.ndarray]
            The positions of the grid in x- and y-direction in cell coordinates

        Returns
        -------
        grid : np.ndarray of shape (len(ranges[0]), len(ranges[1]))
            The grid with the value 1 at all positions in x and y that are covered by the adsorbate directly,
            0 otherwise
        """
        x_range = ranges[0]
        y_range = ranges[1]
        grid = np.zeros(shape=(len(x_range), len(y_range)))
        for xcount, x in enumerate(x_range):
            for ycount, y in enumerate(y_range):
                grid_point = pbc.transform(np.array([x, y, 1.0]))[0]
                xp = grid_point[0]
                yp = grid_point[1]
                if any((a[0] - xp) ** 2 + (a[1] - yp) ** 2 < (adsorbate.vdw_avg) ** 2 + adsorbate.vdw_values[i]
                       for i, a in enumerate(positions)):
                    grid[xcount][ycount] = 1
        return grid

    @staticmethod
    def _fill4(x: int, y: int, xlimit: int, ylimit: int, grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        Apply the fill4 algorithm. A list of visited coordinates is constructed. If the coordinates have already been
        visited, nothing is done. If it is a new point, check the 4 nearest neighbors whether they have been visited

        Parameters
        ----------
        x : int
            The x start coordinate
        y : int
            The y start coordinate
        xlimit : int
            The length of the grid in the x dimension
        ylimit : int
            The length of the grid in the y dimension
        grid : np.ndarray
            The grid

        Returns
        -------
        visit_list : List[Tuple[int, int]]
            All visited indices
        """
        visit_list = []
        todo = [(x, y)]
        while len(todo) > 0:
            xy = todo[-1]
            x = xy[0]
            y = xy[1]
            if x < 0 or y < 0 or x >= xlimit or y >= ylimit:
                # connected fields with zeros are connected to the border --> do not fill
                return []
            elif xy in visit_list:
                # already checked, avoid infinite loops
                del todo[-1]
            elif grid[x][y] == 0:
                # field is zero, visit 4 next neighbours
                visit_list.append(xy)
                del todo[-1]
                todo.append((x + 1, y))
                todo.append((x - 1, y))
                todo.append((x, y + 1))
                todo.append((x, y - 1))
            else:
                # field is 1, do nothing
                visit_list.append(xy)
                del todo[-1]
        return visit_list

    @staticmethod
    def _get_slab_direction(ps: utils.PeriodicSystem, site: np.ndarray) -> np.ndarray:
        """
        Determine the best adsorption direction for the site on the slab based on sterical criterion

        Parameters
        ----------
        ps : utils.PeriodicSystem
            The slab where the adsorbate is placed on
        site : np.ndarray of len 3
            The site on the slab

        Returns
        -------
        slab_direction : np.ndarray of len 3
            The vector representing the best adsorption direction relative to the site
        """
        shift = 20
        # make big slab to minimize artifacts from the finite model
        super_ps = ps * [2 * shift + 1, 2 * shift + 1, 1]
        # ensure that the screened site is centered
        site = ps.pbc.translate_positions_into_cell(site, np.array([shift, shift, 0]))[0]
        # give site the vdw sphere of the closest element
        closest_element, _ = PeriodicUtils.get_closest_element(super_ps.atoms, site)
        elements = super_ps.atoms.elements + [closest_element]
        atom_coords_with_ghost = np.concatenate((super_ps.atoms.positions, np.array([site])))
        # get relative direction for the ghost atom corresponding to site
        inter_generator = InterReactiveComplexes()
        inter_options: Dict[str, Union[int, bool, float]] = {"number_rotamers": 1, "multiple_attack_points": False}
        inter_generator.set_options(inter_options)
        ghost_index = len(super_ps.atoms)
        return inter_generator.get_attack_points_per_atom(atom_coords_with_ghost, elements,
                                                          indices=[ghost_index]
                                                          )[(ghost_index,)][0] - site


class AdsorptionGenerator(InterReactiveComplexes):

    class Options(InterReactiveComplexes.Options):
        """
        The options for the AdsorptionGenerator

        Attributes
        ----------
        wanted_coverage : float
            The percentage of the surface sites that shall be covered by the adsorbate
        adsorption_distance : Union[None, float]
            The distance between the adsorption site and the adsorbing atom of the adsorbate
        maximum_extension : int
            The maximum number to duplicate the slab in x- and y-direction to achieve the right coverage
        eps : float
            The precision to achieve the wanted coverage
        rotamers : bool
            Whether each adsorbed molecule shall be rotated by a random angle
        extension : Union[int, None]
            The number to duplicate the slab in x- and y-direction before placing a single adsorbate in the middle
        check_size : bool
            Whether it should be checked whether the given surface has to be extended to fit the adsorbate
        """

        def __init__(self) -> None:
            super().__init__()
            self.wanted_coverage = 0.25
            self.multiple_molecules = False
            self.adsorption_distance: Optional[float] = None
            self.maximum_extension = 20
            self.eps = 0.01
            self.rotamers = True
            self.extension: Optional[List[int]] = None
            self.check_size = False

    options: Options

    def __init__(self) -> None:
        super().__init__()
        self.options = self.Options()
        self.properties: db.Collection = "required"  # type: ignore

    def generate_reactive_complexes(  # type: ignore  # pylint: disable=arguments-renamed
            self,
            surface: db.Structure,
            adsorbate_db_structure: db.Structure,
            reactive_site_filter: ReactiveSiteFilter = ReactiveSiteFilter()
    ) -> List[AdsorptionResult]:
        if not isinstance(self.properties, db.Collection):
            raise ValueError("This AdsorptionGenerator needs the property collection set to work.")
        surface_atom_indices = PmgInterface.get_surface_atom_indices(surface, self.properties)
        nothing_adsorbed = bool(len(surface_atom_indices) == len(surface.get_atoms()))
        ps = utils.PeriodicSystem(utils.PeriodicBoundaries(surface.get_model().periodic_boundaries),
                                  surface.get_atoms(), surface_atom_indices)
        slab = PmgInterface.to_slab(surface, self.properties)
        n_possible_sites = PeriodicUtils.get_n_top_surface_atoms(ps)
        if abs(ps.pbc.lengths[0] - ps.pbc.lengths[1]) > 0.01:  # n_sites is not quadratic number
            warnings.warn("Got a none quadratic surface. This will most likely result in an error if the generator is "
                          "run in non-specified extension and multiple molecules mode.")
            # this mode is very risky and will crash if the surface is not quadratic, because one cannot find out in
            # which direction the adsorbates have a higher range than the other direction
        finder = AdsorbateSiteFinder(slab)
        sites = finder.find_adsorption_sites() if nothing_adsorbed else finder.find_adsorption_sites(near_reduce=0.2)
        adsorber = Adsorber(slab.as_dict(), sites, wanted_coverage=self.options.wanted_coverage,
                            adsorption_distance=self.options.adsorption_distance,
                            maximum_extension=self.options.maximum_extension,
                            n_surface_atoms=n_possible_sites,
                            eps=self.options.eps)
        n_adsorbate_atoms = len(adsorbate_db_structure.get_atoms())
        atom_indices = reactive_site_filter.filter_atoms([adsorbate_db_structure], list(range(n_adsorbate_atoms)))
        adsorbate = Adsorbate(adsorbate_db_structure, atom_indices)
        if len(adsorbate.atoms) == 1:
            self.options.rotamers = False
        formula = PmgInterface.formula_from_pymatgen_slab(slab)
        if self.options.extension is not None:  # extension is given, so use this function
            try:
                results = adsorber.generate_single_adsorbate_on_given_extension(formula, ps, adsorbate,
                                                                                self.options.extension,
                                                                                self.options.check_size)
            except NotEnoughSpaceInCellException as e:
                if nothing_adsorbed:  # first adsorbate should not fail
                    if not self.options.check_size:
                        # failed because big adsorbate, small extension and not allowed to extend
                        warnings.warn("The adsorbate was too big for the given extension and you did not "
                                      "activate the 'check_size' option. Redoing adsorption with that option.")
                        self.options.check_size = True
                        results = adsorber.generate_single_adsorbate_on_given_extension(formula, ps, adsorbate,
                                                                                        self.options.extension,
                                                                                        self.options.check_size)
                    else:
                        # should not be reached. In this case the NotEnoughSpaceInCellException is probably wrong
                        # the reason for any empty results lies somewhere else
                        raise RuntimeError('Could not place adsorbate. Something unexpected happened.') from e
                else:  # there was already other adsorbate --> need to increase cell
                    warnings.warn('The second adsorbate could not be placed. The reason is probably a too small unit '
                                  'cell. The structure ' + str(surface.get_id()) + ' needs to be extended.')
                    # something for increasing sizes like label or so
                    results = []

        elif nothing_adsorbed:  # other function only works if no adsorbate present
            results = adsorber.generate_adsorption_structures(formula, ps, adsorbate,
                                                              self.options.rotamers, self.options.multiple_molecules)
        else:
            raise NotImplementedError('The adsorption of a molecule onto a surface with another adsorbate solely based '
                                      'on the input of the wanted coverage is not implemented. Please state a wanted '
                                      'extension. If you give a too small number, the surface will be extended to fit '
                                      'the second adsorbate.')
        return results
