#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from collections import Counter
from platform import python_version
from typing import List, Dict, Tuple, Union, Set, Optional
import ast
import numpy as np

import scine_utilities as utils
import scine_database as db
import pymatgen
from pymatgen.ext.matproj import MPRester
from pymatgen.core.surface import generate_all_slabs, SlabGenerator
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import Slab
from polymethod import OverloadMeta, overload


class PmgInterface(metaclass=OverloadMeta):

    @classmethod  # type: ignore
    @overload
    def get_materialsproject_data(cls, formula: str) -> List[dict]:  # noqa: F811
        mpr = MPRester()
        # first get IDs
        ids = mpr.get_materials_ids(formula)  # type: ignore  # pylint: disable=no-member
        if len(ids) == 0:
            raise RuntimeError('Did not find any entries in the MaterialsProject with the formula ' + formula)
        data = []
        for i in ids:
            # get information necessary to decide based on ID
            # direct search not helpful, because we will need the ID
            d = mpr.query(criteria={"material_id": i},  # type: ignore  # pylint: disable=no-member
                          properties=["formation_energy_per_atom", "e_above_hull", "icsd_ids", "nsites"])
            d[0]["id"] = i  # save ID for later
            data.append(d[0])  # query returns list, in this case only one entry since we searched by ID
        return data

    @classmethod  # type: ignore
    @overload
    def get_crystal_structure_by_formula_lowest_sites(  # noqa: F811
            cls,
            formula: str,
            e_cutoff: float = 0.05,
            conventional: bool = True,
            require_experimental: bool = True) -> Structure:
        data = cls.get_materialsproject_data(formula)
        # reduce to lowest energy ones
        data = [datum for datum in data if datum["e_above_hull"] < e_cutoff and
                not require_experimental or datum['icsd_ids']]
        if not data:
            raise RuntimeError('Did not find any entries in the MaterialsProject below the given energy cutoff of ' +
                               str(e_cutoff))
        data = sorted(data, key=lambda x: x["nsites"])
        mpr = MPRester()
        return mpr.get_structure_by_material_id(data[0]["id"],  # type: ignore  # pylint: disable=no-member
                                                conventional_unit_cell=conventional)

    @classmethod  # type: ignore
    @overload
    def get_crystal_structure_by_formula(cls, formula: str, conventional: bool = True,  # noqa: F811
                                         require_experimental: bool = True) -> Structure:
        """
        crawls Materialdatabase for the sought crystal structure
        first finds all IDs corresponding to the entered formula
        then selects the one with the lowest formation energy per atom and records in the ICSD

        Parameters
        ----------
        formula : str
            Chemical formula of the compound such as Ag2O.
        conventional : bool
            Whether the conventional unit cell shall be returned (default)
        require_experimental : bool
        Whether the returned result has to have ICSD records (default)

        Returns
        -------
        crystal : Structure
            A unit cell of the sought crystal
        """
        data = cls.get_materialsproject_data(formula)
        lowest_data: Optional[dict] = None
        sorted_data: List[dict] = sorted(data, key=lambda i: i['formation_energy_per_atom'])
        if require_experimental:
            for d in sorted_data:
                if lowest_data is not None:
                    # if the next lowest formation energy is almost identical, but has more
                    # records in the ICSD -> pick this one
                    if (d['formation_energy_per_atom'] -
                            lowest_data['formation_energy_per_atom']) < 0.01:  # pylint: disable=unsubscriptable-object
                        if len(d["icsd_ids"]) > len(lowest_data["icsd_ids"]):  # pylint: disable=unsubscriptable-object
                            lowest_data = d
                    else:
                        # next lowest has bigger energy or fewer records -> we are finished with the search
                        break
                elif len(d["icsd_ids"]) > 0:
                    # has record in ICSD -> pick this one
                    lowest_data = d
                else:
                    lowest_data = sorted_data[0]
                    if (sorted_data[1]['formation_energy_per_atom'] - lowest_data['formation_energy_per_atom']) < 0.01:
                        raise Warning(
                            'Multiple MaterialsProject entries with very close formation energy per atoms exist. '
                            'We selected ' +
                            lowest_data["id"] +
                            " but " +
                            sorted_data[1]["id"] +
                            ' is also close'
                            ' in energy.')
        if lowest_data is not None:
            # get structure only works with ID
            mpr = MPRester()
            return mpr.get_structure_by_material_id(lowest_data["id"],  # type: ignore  # pylint: disable=no-member
                                                    conventional_unit_cell=conventional)
        else:
            # no result had records in the ICSD -> raise Exception
            raise NameError("No data found")

    @classmethod  # type: ignore
    @overload
    def get_crystal_structure_by_id(cls, id: str, conventional: bool = True) -> Structure:  # noqa: F811
        mpr = MPRester()
        return mpr.get_structure_by_material_id(id,  # type: ignore  # pylint: disable=no-member
                                                conventional_unit_cell=conventional)

    @classmethod  # type: ignore
    @overload
    def get_slabs(cls, crystal: Structure, miller_indices: str, vacuum_separation: float = 15.0,  # noqa: F811
                  min_slab_layers: float = 7.0) -> list:
        """
        generates Slab(s) from crystal structure based on miller indices input
        the input can either be the maximum index or exact miller indices

        Parameters
        ----------
        crystal : Structure
            Crystal structure from which the Slab is cut from
        miller_indices : str
            The wanted miller indices (e.g. "2-11")
        vacuum_separation : float
            The extension of the vacuum above the slab in Angstrom to ensure separation of images
        min_slab_layers : float
            The minimum size of the slab in number of layers

        Returns
        -------
        slabs : List[Slab]
            A list of slabs
        """
        if len(miller_indices) < 3:  # the input is too short
            if len(miller_indices) == 1 and miller_indices.isdigit():
                return cls.get_slabs(crystal, int(miller_indices), vacuum_separation, min_slab_layers)
            raise ValueError("The given miller indices " + miller_indices + " are not valid indices")
        miller = PmgInterface._miller_sanitize(miller_indices)
        gen = SlabGenerator(crystal, miller, min_slab_size=min_slab_layers, min_vacuum_size=vacuum_separation / 4.0,
                            center_slab=True, in_unit_planes=True, max_normal_search=10)
        return gen.get_slabs()

    @classmethod  # type: ignore
    @overload
    def get_slabs(cls, crystal: Structure, max_miller_index: int, vacuum_separation: float = 15.0,  # noqa: F811
                  min_slab_layers: float = 7.0) -> list:
        return generate_all_slabs(crystal, max_index=max_miller_index, min_slab_size=min_slab_layers,
                                  min_vacuum_size=vacuum_separation / 4.0, center_slab=True, in_unit_planes=True,
                                  max_normal_search=10)

    @classmethod  # type: ignore
    @overload
    def get_slabs(cls, crystal: Structure, miller_indices: list, vacuum_separation: float = 15.0,  # noqa: F811
                  min_slab_layers: float = 7.0) -> list:
        gen = SlabGenerator(crystal, miller_indices, min_slab_size=min_slab_layers,
                            min_vacuum_size=vacuum_separation / 4.0, center_slab=True, in_unit_planes=True,
                            max_normal_search=10)
        return gen.get_slabs()

    @classmethod
    def _miller_sanitize(cls, miller: str) -> List[int]:
        """
        cleans up string input into list which can be parsed to SlabGenerator

        Parameters
        ----------
        miller : str
            miller indices (e.g. "2-11")

        Returns
        -------
        miller_indices : List[int]
            A list of the miller index
        """
        miller_indices = []
        negative = False
        for m in miller:
            if m == "-":
                # determine negative integers
                negative = True
            elif negative:
                miller_indices.append(-1 * int(m))
                negative = False
            else:
                miller_indices.append(int(m))

        # remove redundant 4th bravais
        if len(miller_indices) == 4:
            del (miller_indices[2])
        assert len(miller_indices) == 3
        return miller_indices

    @classmethod
    def mat_sites_to_bohr(cls, sites: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        converts coordinates of sites from pymatgen into Bohr

        Parameters
        ----------
        sites : Dict[str, np.ndarray of shape (n,3)]
            sites given by name of their symmetry site and then list of site coordinates

        Returns
        -------
        new_sites : Dict[str, np.ndarray]
            Sites with Bohr coordinates
        """
        return_sites: Dict[str, np.ndarray] = {}
        for key, value in sites.items():
            return_sites[key] = np.array(value) * utils.BOHR_PER_ANGSTROM
        return return_sites

    @classmethod
    def to_pbc(cls, slab: Slab) -> utils.PeriodicBoundaries:
        return utils.PeriodicBoundaries(slab.lattice.matrix * utils.BOHR_PER_ANGSTROM)

    @classmethod
    def to_string(cls, slab: Slab) -> str:
        """
        converts coordinates of sites from pymatgen into Bohr

        Parameters
        ----------
        slab : Slab
            The slab you want the pbc from

        Returns
        -------
        pbc : str
            The periodic boundary conditions (a, b, c, alpha, beta, gamma) as string
        """
        pbc = cls.to_pbc(slab)
        return str(pbc)

    @classmethod  # type: ignore
    @overload
    def write_cif(cls, structure: db.Structure, db_structures: db.Collection, filename: str) -> None:
        """
        writes database structure with periodic boundary condition in model to .cif file

        Parameters
        ----------
        structure : db.Structure
            Structure in database part of collection of structures
        db_structures : db.Collection
            Collection of structures in database
        filename : str
            Name of the .cif file
        """
        structure.link(db_structures)
        cls.write_cif(structure, filename)

    @classmethod  # type: ignore
    @overload
    def write_cif(cls, structure: db.Structure, filename: str) -> None:  # noqa: F811
        """
        writes database structure with periodic boundary condition in model to .cif file

        Parameters
        ----------
        structure : db.Structure
            Structure in database part of collection of structures
        filename : str
            Name of the .cif file
        """
        ac = structure.get_atoms()
        pbc = structure.get_model().periodic_boundaries
        if '.cif' not in filename:
            filename += '.cif'
        cls.write_cif(ac, pbc, filename)

    @classmethod  # type: ignore
    @overload
    def write_cif(cls, ac: utils.AtomCollection, pbc: str, filename: str) -> None:  # noqa: F811
        """
        writes AtomCollection to .cif file corresponding to provided periodic boundary condition

        Parameters
        ----------
        ac : utils.AtomCollection
            Atoms to be written to .cif file
        pbc : str
            periodic boundary conditions as 'a,b,c,alpha,beta,gamma'
        filename : str
            Name of the .cif file
        """
        cell = utils.PeriodicBoundaries(pbc)
        cls.write_cif(ac, cell, filename)

    @classmethod  # type: ignore
    @overload
    def write_cif(cls, system: utils.PeriodicSystem, filename: str) -> None:  # noqa: F811
        cls.write_cif(system.atoms, system.pbc, filename)

    @classmethod  # type: ignore
    @overload
    def write_cif(cls, ac: utils.AtomCollection, pbc: utils.PeriodicBoundaries, filename: str) -> None:  # noqa: F811
        ele = [utils.ElementInfo.symbol(e) for e in ac.elements]
        coords = utils.ANGSTROM_PER_BOHR * ac.positions
        struc = Structure(pbc.matrix, ele, coords, coords_are_cartesian=True)
        writer = CifWriter(struc)
        writer.write_file(filename)

    @classmethod  # type: ignore
    @overload
    def to_mat_structure(cls, slab: Slab, atoms: utils.AtomCollection) -> Structure:  # noqa: F811
        """
        get pymatgen Structure from slab and AtomCollection
        slab determines the defined periodic boundary conditions
        atoms determine the included atoms in the structure

        Parameters
        ----------
        slab : Slab
            The slab you want the lattice from
        atoms : utils.AtomCollection
            The atoms written to the structure

        Returns
        -------
        structure : Structure
            The pymatgen Structure
        """
        angstrom_pos = utils.ANGSTROM_PER_BOHR * atoms.positions
        return Structure(slab.lattice, [str(e) for e in atoms.elements], angstrom_pos, coords_are_cartesian=True)

    @classmethod  # type: ignore
    @overload
    def to_mat_structure(cls, ps: utils.PeriodicSystem) -> Structure:  # noqa: F811
        """
        get pymatgen Structure from PeriodicSystem

        Parameters
        ----------
        ps : utils.PeriodicSystem
            The PeriodicSystem you want a pmg Structure from

        Returns
        -------
        structure : Structure
            The pymatgen Structure
        """
        angstrom_pos = utils.ANGSTROM_PER_BOHR * ps.atoms.positions
        lattice = utils.ANGSTROM_PER_BOHR * ps.pbc.matrix
        return Structure(lattice, [str(e) for e in ps.atoms.elements], angstrom_pos, coords_are_cartesian=True)

    @classmethod
    def get_slab_dict(cls, structure: db.Structure, properties: db.Collection) -> dict:
        if not structure.has_property("slab_dict"):
            raise RuntimeError('Slab information is missing for structure ' + str(structure))
        dict_info = db.StringProperty(structure.get_property('slab_dict'), properties)
        return cls.slab_dict_string_to_dict(dict_info.get_data())

    @classmethod
    def slab_dict_string_to_dict(cls, slab_dict_string: str) -> dict:
        dict_info_string = slab_dict_string  # copy so the input string is not altered
        # remove some specific extra strings from representation to be able to make dict out of string
        dict_info_string = dict_info_string.replace("]])", "]]")
        dict_info_string = dict_info_string.replace("array(", "")
        # transform into dict
        return ast.literal_eval(dict_info_string)

    @classmethod  # type: ignore
    @overload
    def to_slab(cls, structure: db.Structure, properties: db.Collection) -> Slab:  # noqa: F811
        """
        get the pymatgen Slab object from a database structure
        the database structure needs to have the 'slab_dict' property
        Watch out: The coordinates of the slab sites are translated relative to atoms of db.Structure
        -> fit before doing operations involving both

        Parameters
        ----------
        structure : db.Structure
            The database structure that represent a surface slab
        properties : db.Collection
            The properties collection of the database

        Returns
        -------
        slab : Slab
            The pymatgen Slab
        """
        return Slab.from_dict(cls.get_slab_dict(structure, properties))

    @classmethod  # type: ignore
    @overload
    def to_slab(cls, slab_dict: dict) -> Slab:  # noqa: F811
        return Slab.from_dict(slab_dict)

    @classmethod  # type: ignore
    @overload
    def to_slab(cls, slab: Slab, ps: utils.PeriodicSystem) -> Slab:  # noqa: F811
        return cls.to_slab(slab, ps.atoms, ps.pbc)

    @classmethod  # type: ignore
    @overload
    def to_slab(cls, slab: Slab, ac: utils.AtomCollection) -> Slab:  # noqa: F811
        pbc = cls.to_pbc(slab)
        return cls.to_slab(slab, ac, pbc)

    @classmethod  # type: ignore
    @overload
    def to_slab(cls, slab: Slab, ac: utils.AtomCollection, pbc: utils.PeriodicBoundaries) -> Slab:  # noqa: F811
        ele = [utils.ElementInfo.symbol(e) for e in ac.elements]
        coords = pbc.transform(ac.positions, False)
        lattice = cls._construct_pmg_lattice(pbc)
        new_slab = Slab(lattice, ele, coords, slab.miller_index, slab.oriented_unit_cell, slab.shift,
                        slab.scale_factor, coords_are_cartesian=False)
        return new_slab

    @classmethod
    def get_surface_atom_indices(cls, structure: db.Structure, properties: db.Collection) -> Set[int]:
        if not structure.has_property('surface_atom_indices'):
            raise RuntimeError('The indices of surface atoms are missing as a property of the surface structure.')
        surface_atoms_prop = db.VectorProperty(structure.get_property('surface_atom_indices'))
        surface_atoms_prop.link(properties)
        return set([int(i) for i in surface_atoms_prop.get_data()])

    @classmethod
    def to_db_structure(cls, slab: Slab, structures: db.Collection, properties: db.Collection,
                        label: db.Label, model: db.Model, charge: int, multiplicity: int):
        model.periodic_boundaries = cls.to_string(slab)
        ac = cls.to_atomcollection(slab)
        structure = db.Structure()
        structure.link(structures)
        structure.create(ac, charge, multiplicity, model, label)

        slab_property = db.StringProperty()
        slab_property.link(properties)
        slab_property.create(model, 'slab_dict', str(slab.as_dict()))
        structure.set_property('slab_dict', slab_property.get_id())

        return structure

    @classmethod
    def to_atomcollection(cls, slab: Slab) -> utils.AtomCollection:
        elements = []
        coords = []
        for site in slab.sites:
            elements.append(utils.ElementInfo.element_from_symbol(site.species_string))
            coords.append(site.coords)
        coords_array = np.asarray(coords) * utils.BOHR_PER_ANGSTROM
        return utils.AtomCollection(elements, coords_array)

    @classmethod  # type: ignore
    @overload
    def to_periodic_system(cls, slab: Slab, surface_atom_indices=None) -> utils.PeriodicSystem:  # noqa: F811
        ac = cls.to_atomcollection(slab)
        pbc = cls.to_pbc(slab)
        if surface_atom_indices is None:
            return utils.PeriodicSystem(pbc, ac)
        return utils.PeriodicSystem(pbc, ac, set(surface_atom_indices))

    @classmethod  # type: ignore
    @overload
    def to_periodic_system(cls, structure: db.Structure,  # noqa: F811
                           properties: Optional[db.Collection] = None) -> utils.PeriodicSystem:
        ac = structure.get_atoms()
        pbc_string = structure.get_model().periodic_boundaries
        if not pbc_string or pbc_string.lower() == "none":
            raise RuntimeError(f"Given structure {str(structure.id())} does not have periodic boundaries")
        pbc = utils.PeriodicBoundaries(pbc_string)
        if properties is not None and structure.has_property("surface_atom_indices"):
            indices_prop = db.VectorProperty(structure.get_property("surface_atom_indices"), properties)
            indices = set(indices_prop.get_data())
        else:
            indices = set()
        return utils.PeriodicSystem(pbc, ac, indices)

    @classmethod
    def formula_from_pymatgen_slab(cls, slab: Slab) -> str:
        """
        based on the elements and their numbers present in a slab, a chemical formula is generated
        The least abundant element is set to 1 in the formula and the other elements are multiples of that
        In the formula 1's are not explicitly written
        The formula is NOT written in the IUPAC standard, but alphabetically

        Parameters
        ----------
        slab : Slab
            The pymatgen Slab

        Returns
        -------
        formula : str
            The chemical formula of the slab
        """
        elements = [str(e) for e in slab.species]
        counter = Counter(elements)
        # minimum value is set to one in new dict called reduced
        m = min(counter.values())
        reduced = {}
        for key, value in counter.items():
            reduced[key] = value / m
        # sort elements to have some consistency, even if not to IUPAC standard
        sorted_elements = sorted(counter.keys())
        formula = ''
        for ele in sorted_elements:
            num = reduced[ele]
            if any(abs(num - i) < 1e-6 for i in range(100)):  # number is integer
                num = int(round(num))
                formula += ele
                if num != 1:
                    formula += str(num)
            elif any(abs(num - i) < 1e-6 for i in np.arange(0.5, 100.0, 1.0)):  # number is .5 float
                formula += ele + "{0:.1f}".format(num)
            else:
                formula += ele + "{0:.2f}".format(num)
        return formula

    @classmethod
    def insert_slabs(cls,
                     slabs: List[Slab],
                     charge: int,
                     multiplicity: int,
                     model: db.Model,
                     label: db.Label,
                     job: db.Job,
                     settings: Optional[dict],
                     extension: int,
                     manager: db.Manager) -> List[Tuple[db.Structure,
                                                        db.Calculation]]:
        structures = manager.get_collection('structures')
        calculations = manager.get_collection('calculations')
        properties = manager.get_collection('properties')

        results: List[Tuple[db.Structure, db.Calculation]] = []
        for slab in slabs:
            extension_list = [extension, extension, 1]
            super_slab, _ = cls.to_superslab(slab, extension_list)
            structure = cls.to_db_structure(super_slab, structures, properties, label, model, charge,
                                            multiplicity)

            cls._set_properties_of_new_slab(structure, slab, super_slab, properties, model)

            calculation = db.Calculation(db.ID(), calculations)
            model.periodic_boundaries = structure.get_model().periodic_boundaries
            calculation.create(model, job, [structure.id()])
            if settings is not None:
                calculation.set_settings(utils.ValueCollection(settings))
            calculation.set_status(db.Status.HOLD)
            results.append((structure, calculation))
        return results

    @classmethod  # type: ignore
    @overload
    def _set_properties_of_new_slab(  # noqa: F811
            cls,
            structure: db.Structure,
            slab: Slab,
            super_slab: Slab,
            properties: db.Collection,
            model: db.Model):
        formula_prop = db.StringProperty(db.ID(), properties)
        formula_prop.create(model, 'slab_formula', cls.formula_from_pymatgen_slab(slab))
        structure.set_property('slab_formula', formula_prop.get_id())

        surf_atoms_prop = db.VectorProperty(db.ID(), properties)
        surf_atoms_prop.create(model, 'surface_atom_indices', np.array(list(range(len(super_slab.sites)))))
        structure.set_property('surface_atom_indices', surf_atoms_prop.get_id())

        primitive_lattice_prop = db.DenseMatrixProperty(db.ID(), properties)
        # pymatgen has different lattice matrix convention and uses angstrom
        primitive_lattice_prop.create(model, 'primitive_lattice', slab.lattice.matrix.T * utils.BOHR_PER_ANGSTROM)
        structure.set_property('primitive_lattice', primitive_lattice_prop.get_id())

    @classmethod  # type: ignore
    @overload
    def update_slab_dict(cls, structure: db.Structure, properties: db.Collection) -> None:  # noqa: F811
        slab_dict = cls.get_slab_dict(structure, properties)
        ps = cls.to_periodic_system(structure, properties)
        new_slab_dict = cls.update_slab_dict(slab_dict, ps)
        slab_property = db.StringProperty(db.ID(), properties)
        slab_property.create(structure.get_model(), 'slab_dict', str(new_slab_dict))
        if structure.has_property('slab_dict'):
            structure.clear_properties('slab_dict')
        structure.set_property('slab_dict', slab_property.id())

    @classmethod  # type: ignore
    @overload
    def update_slab_dict(cls, slab_dict: dict, ps: utils.PeriodicSystem) -> dict:  # noqa: F811
        slab = Slab.from_dict(slab_dict)
        new_slab = cls.to_slab(slab, ps)
        return new_slab.as_dict()

    @classmethod  # type: ignore
    @overload
    def update_slab_sites(cls, slab: Slab, atoms: utils.AtomCollection) -> Slab:  # noqa: F811
        if len(slab.sites) != len(atoms):
            raise RuntimeError(f"Slab {slab}\nand atoms {atoms.elements}\n{atoms.positions}\nhave different sizes")
        struc = slab * 1
        ele = [s for s in struc.species]
        sites = [s for s in struc.sites]
        coords = np.array([atoms.get_position(i) * utils.ANGSTROM_PER_BOHR for i in range(len(sites))])
        new_slab = Slab(struc.lattice, ele, coords, slab.miller_index, slab.oriented_unit_cell, slab.shift,
                        slab.scale_factor, coords_are_cartesian=True)
        return new_slab

    @classmethod
    def to_superslab(cls, slab: Slab, extension: List[int],
                     surface_atom_indices: Union[Set[int], List[int], None] = None) \
            -> Tuple[Slab, Optional[List[int]]]:
        """
        return Slab object which is a super slab of the passed slab
        the standard multiplication would return an Structure object, which misses some important Slab information

        Parameters
        ----------
        slab : Slab
            The slab
        extension : Optional[List[int]]
            The extension in x, y, and z direction. The default is [2, 2, 1]
        surface_atom_indices : Optional[List[int]]
            A list of all atom indices that are part of the actual surface

        Returns
        -------
        superslab, new_indices : Tuple[Slab, Optional[List[int]]
            the extended slab, List of all atom indices part of surface after extension
        """
        struc_superslab = slab * extension
        ele = [s for s in struc_superslab.species]
        sites = [s for s in struc_superslab.sites]
        coords = np.array([site.coords for site in sites])
        superslab = Slab(struc_superslab.lattice, ele, coords, slab.miller_index, slab.oriented_unit_cell, slab.shift,
                         slab.scale_factor, coords_are_cartesian=True)
        if surface_atom_indices is None:
            return superslab, None
        ps = cls.to_periodic_system(slab, surface_atom_indices)
        ps *= np.asarray(extension)
        return superslab, list(ps.solid_state_atom_indices)

    @classmethod
    def _construct_pmg_lattice(cls, pbc: utils.PeriodicBoundaries) -> Lattice:
        matrix = pbc.matrix * utils.ANGSTROM_PER_BOHR
        if python_version() >= "3.8":
            from importlib.metadata import version
            pymatgen_version = version(pymatgen.__name__)
        elif hasattr(pymatgen, "__version__"):
            pymatgen_version = getattr(pymatgen, "__version__")
        else:
            pymatgen_version = "0"
        if pymatgen_version >= "2022":
            return Lattice(matrix, pbc.periodicity)  # type: ignore  # pylint: disable=too-many-function-args
        return Lattice(matrix)
