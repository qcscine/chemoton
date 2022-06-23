#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import json
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from json import dumps
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Third party imports
import scine_utilities as utils
import scine_database as db
import scine_molassembler as masm

import puffin.utilities.masm_helper as masm_helper


class Toolbox:
    def __init__(
        self,
        structures: db.Collection,
        calculations: db.Collection,
        compounds: db.Collection,
        properties: db.Collection,
        elementary_steps: db.Collection,
        reactions: db.Collection,
    ):
        """
        Write all collections of a database into variables.
        """

        # # # Get the collctions in the database
        self._structures = structures
        self._calculations = calculations
        self._compounds = compounds
        self._properties = properties
        self._elementary_steps = elementary_steps
        self._reactions = reactions

        # # # Setup Plots
        self._setup_plt()

    def display_masm_cbor_graph(self, masm_cbor_str: str):
        """
        Display molassembler CBOR string as graph.

        Parameter
        ----------
        masm_cbor_str :: str
            Graph encoded as cbor string.
        """

        # convert str in masm_cbor to graph representation
        tmp_binary = masm.JsonSerialization.base_64_decode(masm_cbor_str)
        tmp_str_2 = masm.JsonSerialization(tmp_binary, masm.JsonSerialization.BinaryFormat.CBOR)
        masm_molecule = tmp_str_2.to_molecule()
        display.display_svg(masm_molecule)

    def query_structures_by_mol_references(self, rel_path: str) -> Dict[str, db.ID]:
        """
        Find molecules given as '.mol' files in the directory given as relative
        path. Returns the found compounds as dictionary with the
        'masm_cbor_graph' string as key.
        The '.mol' files has to contain additional information about the charge
        and the multiplicity of the molecule, given in the third line of the
        '.mol' file in the form of '{ "charge": 0, "multiplicity": 1 }'..

        Parameter
        ----------
        rel_path :: str
            The relative path to the directory containing the '.mol' files.
        Returns
        -------
        found_compounds :: Dict[str, db.ID]
            A dictionary with the 'masm_cbor_graph' as key for every entry of
            the compound ID.
        """

        selections, mol_files = self._convert_mol_files_to_selections(rel_path)
        checked_compounds_str = []

        found_compounds = {}

        # # # Loop over seletions corresponding to mol files
        for index, selection in enumerate(selections):
            # # # Loop over structures fitting the selection
            for tmp_structure in self._structures.iterate_structures(selection):
                tmp_structure.link(self._structures)
                tmp_compound_id = tmp_structure.get_compound()
                tmp_compound_str = tmp_compound_id.string()
                if tmp_compound_str not in checked_compounds_str:
                    tmp_structure_graph = tmp_structure.get_graph("masm_cbor_graph")
                    checked_compounds_str.append(tmp_compound_str)
                    found_compounds[tmp_structure_graph] = {
                        "cmp_id": tmp_compound_id,
                        "ref": mol_files[index],
                    }

        return found_compounds

    def get_reaction_profile(
        self,
        reaction_id: db.ID,
        property_key: str = "gibbs_free_energy",
        path_mode: str = "min",
    ) -> Tuple[np.ndarray, str]:
        """
        Returns an array containing the barriers in the forward and backward
        direction as well as the reaction energy of a given reaction. The
        energies are either free energies or, if these are not available for at
        least one reactant (eg. one single atom), electronic energies. The
        barriers can be deduced from the transition state with the minimal or
        maximal energie or the mean over all found transition states.

        Parameter
        ----------
        reaction_id :: db.ID
            The ID of the reaction of interest
        property_key :: str
            The property_key of the energy of interest. Per default set to
            'gibbs_free_energy'.
        path_mode :: str
            The mode of the path, meaning via which TS respectively elementary
            step the barriers are calculated. Per default set to 'min', meaning
            the TS with the minimum energy is used for calculating barriers.
            Other modes are 'mean', where the mean energy over all TS is used,
            and 'max', where the TS with the maximum energy is used.

        Returns
        -------
        energy_array :: numpy.ndarray(3, 1)
            Array of shape (3, 1), containing the forward (1, 1), backward
            barrier (2, 1) and the reaction energy
            (3, 1).
        mode :: str
            The 'property_key' of the energy used to calculate the barriers.
        """
        reaction = self._reactions.get_reaction(reaction_id)
        reaction.link(self._reactions)

        # # # Analyze all transition states of Reaction
        ts_energy_list = list()
        ts_energy_list_bck = list()

        # # # Loop over all TSs and store their energies
        for ele_step_id in reaction.get_elementary_steps():
            ele_step = self._elementary_steps.get_elementary_step(ele_step_id)
            ele_step.link(self._elementary_steps)
            transition_state = ele_step.get_transition_state()
            ts_structure = self._structures.get_structure(transition_state)
            ts_energy_list.append(self._get_energy(ts_structure, property_key))
            ts_energy_list_bck.append(self._get_energy(ts_structure, "electronic_energy"))

        # # # Chose TS based on path mode
        no_gibbs = False
        if all(energy is None for energy in ts_energy_list):
            no_gibbs = True
        if path_mode == "mean":
            if not no_gibbs:
                ts_energy = np.mean(ts_energy_list)
            ts_energy_bck = np.mean(ts_energy_list_bck)
        elif path_mode == "max":
            if not no_gibbs:
                ts_energy = np.max(ts_energy_list)
            ts_energy_bck = np.max(ts_energy_list_bck)
        else:
            if not no_gibbs:
                ts_energy = np.min(ts_energy_list)
            ts_energy_bck = np.min(ts_energy_list_bck)

        # # # Analyze both sides of a reaction
        reactants = reaction.get_reactants(db.Side.BOTH)
        reactants_energies = list()
        reactants_energies_bck = list()

        # # # Loop over all sides and all reactants on one side; store the energies
        for side in reactants:
            side_energies = list()
            side_energies_bck = list()
            for reagent in side:
                tmp_c = self._compounds.get_compound(reagent)
                tmp_c.link(self._compounds)
                tmp_s_id = tmp_c.get_centroid()
                tmp_s = self._structures.get_structure(tmp_s_id)
                tmp_energy = self._get_energy(tmp_s, property_key)
                tmp_energy_bck = self._get_energy(tmp_s, "electronic_energy")
                side_energies.append(tmp_energy)
                side_energies_bck.append(tmp_energy_bck)

            reactants_energies.append(side_energies)
            reactants_energies_bck.append(side_energies_bck)

        # # # Check, if the electronic energy or the property_key energy should be used
        # # # Determine the forward and backward barrier as well as the reaction energy
        if no_gibbs or None in reactants_energies[0] or None in reactants_energies[1]:
            forward = ts_energy_bck - np.sum(reactants_energies_bck[0])
            backward = ts_energy_bck - np.sum(reactants_energies_bck[1])
            # # # rhs minus lhs for reaction energy
            equilibrium = np.sum(reactants_energies_bck[1]) - np.sum(reactants_energies_bck[0])
            mode = "electronic_energy"
        else:
            forward = ts_energy - np.sum(reactants_energies[0])
            backward = ts_energy - np.sum(reactants_energies[1])
            # # # rhs minus lhs for reaction energy
            equilibrium = np.sum(reactants_energies[1]) - np.sum(reactants_energies[0])
            mode = property_key

        energy_array = np.asarray([[forward], [backward], [equilibrium]])

        return energy_array, mode

    def get_calculation_status(self) -> Dict[str, int]:
        """
        Returns the count of calculations with different status.

        Returns
        -------
        current_status :: Dict[str, int]
            A dictionary containing the counts of the different states of the
            status.
        """

        available_status = [
            "analyzed",
            "complete",
            "construction",
            "failed",
            "hold",
            "new",
            "pending",
        ]
        current_status = {}

        for status in available_status:
            current_status[status] = self._calculations.count(dumps({"status": {"$eq": status}}))

        # # # Return calculation status
        return current_status

    def get_reaction_equation(self, reaction_id: db.ID) -> Union[str, None]:
        """
        Returns the reaction equation as string. Structures are summarized into
        molecular formulas. Charge and multiplicity are given as well.

        Parameter
        ----------
        reaction_id :: db.ID
            The reaction of interest

        Returns
        -------
        rxn_eq :: str
            The reaction equation represented as string.

        """
        # # # Check, if reaction exists
        if not self._reactions.has_reaction(reaction_id):
            print(
                5 * "!",
                "Reaction " + reaction_id.string() + " not in database",
                5 * "!",
            )

            return None

        reaction = self._reactions.get_reaction(reaction_id)
        reaction.link(self._reactions)
        elementary_step = self._elementary_steps.get_elementary_step(reaction.get_elementary_steps()[0])

        elementary_step.link(self._elementary_steps)
        reactants = elementary_step.get_reactants(db.Side.BOTH)
        rxn_eq = ""
        # # # Loop over lhs and rhs
        for side in range(0, 2):
            for tmp_struct_id in reactants[side]:
                tmp_struct = self._structures.get_structure(tmp_struct_id)
                tmp_struct.link(self._structures)
                tmp_atoms = tmp_struct.get_atoms()
                # # # Add chemical formula
                rxn_eq += utils.generate_chemical_formula(tmp_atoms.elements)
                # # # Add charge
                rxn_eq += "(c:" + str(tmp_struct.get_charge())
                # # # Add multiplicity
                rxn_eq += ", m:" + str(tmp_struct.get_multiplicity()) + ")"
                # # # Add plus sign for addition
                rxn_eq += " + "
            # remove last added plus sign
            rxn_eq = rxn_eq[:-3]
            if side == 0:
                rxn_eq += " = "

        return rxn_eq

    def get_job_distribution(self, status: str) -> Dict[str, int]:
        """
        Returns the distribution of calculations with given state over the
        available jobs.

        Parameter
        ----------
        status :: str
            The status of interest

        Returns
        -------
        current_distribution :: Dict[str, int]
            A dictionary containing the counts of the different jobs in
            calculations.

        """

        available_jobs = [
            "conformers",
            "graph",
            "scine_afir",
            "scine_bond_orders",
            "scine_geometry_optimization",
            "scine_hessian",
            "scine_irc_scan",
            "scine_react_complex_afir",
            "scine_react_complex_nt",
            "scine_single_point",
            "scine_ts_optimization",
            "sleep",
        ]

        current_distribution = {}
        for job in available_jobs:
            selection = {"$and": [{"status": {"$eq": status}}, {"job.order": {"$eq": job}}]}

            current_distribution[job] = self._calculations.count(json.dumps(selection))

        return current_distribution

    def plot_overview_calculations(self, bin_size: float = 1.0):
        """
        Plot a histogram of created and touched calculations.

        Parameter
        ----------
        bin_size :: float
            Size per bin in hours, the default is 1.0
        """
        timestamps_created = list()
        timestamps_last_mod = list()

        # # # Collect timestamps of all calculations
        for tmp_calc in self._calculations.iterate_calculations(json.dumps({"status": {"$eq": "complete"}})):
            tmp_calc.link(self._calculations)
            timestamps_created.append(tmp_calc.created())
            timestamps_last_mod.append(tmp_calc.last_modified())

        # # # One bin per hour
        number_bins = int((max(timestamps_last_mod) - min(timestamps_created)).total_seconds() / (bin_size * 3600))

        # # # Setup subplot
        fig, ax = plt.subplots(figsize=(12, 6))

        plt.hist(
            [timestamps_created, timestamps_last_mod],
            bins=number_bins,
            range=(min(timestamps_created), max(timestamps_last_mod)),
            color=["lightblue", "lightgreen"],
            stacked=False,
        )
        plt.legend(["# Created", "# Last Modified"])

        ax = self._setup_ax(ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Counts")
        plt.show()

        return ax

    def plot_overview_compounds(self, bin_size: float = 1.0):
        """
        Plot a histogram of created and touched compounds.

        Parameter
        ----------
        bin_size :: float
            Size per bin in hours, the default is 1.0
        """
        timestamps_created = list()
        timestamps_last_mod = list()

        # # # Collect timestamps of all compounds
        for tmp_comp in self._compounds.iterate_all_compounds():
            tmp_comp.link(self._compounds)
            timestamps_created.append(tmp_comp.created())
            timestamps_last_mod.append(tmp_comp.last_modified())

        # # # One bin per hour
        number_bins = int((max(timestamps_last_mod) - min(timestamps_created)).total_seconds() / (bin_size * 3600))
        if number_bins < 10:
            number_bins = 10
        # # # Print current number of compounds
        print(
            "Number of Compounds:",
            self._compounds.count(dumps({"structures": {"$exists": True}})),
        )

        # # # Setup subplot
        fig, ax = plt.subplots(figsize=(12, 6))

        plt.hist(
            [timestamps_created, timestamps_last_mod],
            bins=number_bins,
            range=(min(timestamps_created), max(timestamps_last_mod)),
            color=["lightblue", "lightgreen"],
            stacked=False,
        )
        plt.legend(["# Created", "# Last Modified"])
        ax = self._setup_ax(ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Counts")
        plt.show()

    def plot_nt_steps(self, bin_number: float = 20):
        """
        Plot histogram of steps in Newton-Trajectory job of complete (jobs which
        led to different start and end structure) jobs.

        Parameter
        ----------
        bin_number :: float
            Number of bins, the default is 20
        """
        # # # List containing steps of Newton Trajectory jobs
        nt_steps = list()
        start_pattern = "Converged after"
        end_pattern = "iterations"
        # # # Selection iteration over calculations
        selection = {
            "$and": [
                {"status": {"$eq": "complete"}},
                {"comment": {"$ne": "NT Job: No TS guess found."}},
                {"job.order": {"$eq": "scine_react_complex_nt"}},
            ]
        }
        for tmp_calc in self._calculations.iterate_calculations(dumps(selection)):
            tmp_calc.link(self._calculations)
            tmp_output = tmp_calc.get_raw_output()
            tmp_start = tmp_output.find(start_pattern) + len(start_pattern) + 1
            tmp_end = tmp_output[tmp_start:].find(end_pattern) + tmp_start
            nt_steps.append(int(tmp_output[tmp_start:tmp_end]))

        # # # Setup subplot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = self._setup_ax(ax)
        plt.hist(nt_steps, bins=bin_number, color="lightblue", stacked=False)
        plt.show()

    def query_structures_by_smiles(self, smile: str, charge: int = 0, multiplicity: int = 1) -> Dict[str, List[db.ID]]:
        """
        Find compounds which have at least the same bonding situation (same
        elements and bonds) or even the same shape or are completely identical.

        Parameter
        ----------
        smile :: str
            SMILE representation of the molecule as string.
        charge :: int
            Charge of the molecule of interest.
        multiplicity :: int
            Multiplicity (2*S+1) of the molecule of interest.

        Returns
        -------
        fitting_compounds :: Dict[str, List[db.ID]]
            A dictionary differing between the bonding situation
            ('elements_bonds'), the shape ('shape') and the completely identical
            configuration ('identical'), containing a list of compound IDs for
            each entry.
        """
        # # # Interpret given smiles
        query_masm_molecule = masm.io.experimental.from_smiles(smile)
        query_masm_molecule_cbor = masm.JsonSerialization(query_masm_molecule).to_binary(
            masm.JsonSerialization.BinaryFormat.CBOR
        )
        query_masm_molecule_str = masm.JsonSerialization.base_64_encode(query_masm_molecule_cbor)
        # # # Selection for querying structures
        selection = {
            "$and": [
                {"label": {"$in": ["minimum_optimized", "user_optimized"]}},
                {"nAtoms": {"$eq": query_masm_molecule.graph.V}},
                {"charge": {"$eq": charge}},
                {"multiplicity": {"$eq": multiplicity}},
                {"properties.electronic_energy": {"$exists": "true"}},
                {"properties.gibbs_energy_correction": {"$exists": "true"}},
            ]
        }

        checked_compounds = []

        fitting_coumpounds = {"elements_bonds": [], "shape": [], "identical": []}

        for tmp_structure in self._structures.iterate_structures(dumps(selection)):
            tmp_structure.link(self._structures)
            # # # Get compound ID
            tmp_compound_id = tmp_structure.get_compound()
            tmp_compound_str = tmp_compound_id.string()
            # # # Get graph of structure and its masm molecule
            tmp_graph_str = tmp_structure.get_graph("masm_cbor_graph")
            tmp_graph_binary = masm.JsonSerialization.base_64_decode(tmp_graph_str)
            tmp_masm_molecule = masm.JsonSerialization(
                tmp_graph_binary, masm.JsonSerialization.BinaryFormat.CBOR
            ).to_molecule()
            result_key = None
            # # # Check if structure has been tested already
            if tmp_compound_str not in checked_compounds:
                checked_compounds.append(tmp_compound_str)
                # # # Check if structure has same elements and bonds
                if (
                    query_masm_molecule.modular_isomorphism(
                        tmp_masm_molecule,
                        masm.AtomEnvironmentComponents.ElementsAndBonds,
                    )
                    is not None
                ):
                    result_key = "elements_bonds"
                    # # # Check if structure has same shape
                    if (
                        query_masm_molecule.modular_isomorphism(
                            tmp_masm_molecule,
                            masm.AtomEnvironmentComponents.ElementsBondsAndShapes,
                        )
                        is not None
                    ):
                        result_key = "shape"
                        # # # Check if structure is exactly the same
                        if query_masm_molecule_str == tmp_graph_str:
                            result_key = "identical"
            # # # Assign compound ID to fitting category
            if result_key is not None:
                fitting_coumpounds[result_key].append(tmp_compound_id)

        return fitting_coumpounds

    def _convert_mol_files_to_selections(self, structure_folder: str) -> List[str]:
        """
        Reads in the mol files and returns the selection strings to query for to
        find equivalent structures.

        Equivalent structures have to have the same graphs (masm.cbor), charge
        and multiplicity.

        Parameters
        ----------
        structure_folder :: str
            The folder containing the product mol-files.

        Returns
        -------
        List[str]
            The selections to query for.
        """

        selections = []
        mol_files = []

        for mol_file in Path(structure_folder).rglob("*.mol"):
            # # # Charge and multiplicity from comment line
            atoms, bonds = utils.io.read(mol_file.as_posix())
            with open(mol_file, "r") as mol_file:
                # # # Charge and multiplicity from comment line of mol file
                charge_mult = json.loads(mol_file.readlines()[2])

            # # # Utils.IO generates an empty bond order collection for monoatomic systems
            # # # which would result in dimensionality problems later on
            if atoms.size() == 1:
                bonds = utils.BondOrderCollection(1)
            # # # Get graph and decision list
            # # # Use masm_helper to achieve analogy with chemoton
            product_graphs = masm_helper.get_cbor_graph(atoms, bonds, distance_connectivity=True)
            selection = {
                "$and": [
                    {"charge": {"$eq": charge_mult["charge"]}},
                    {"multiplicity": {"$eq": charge_mult["multiplicity"]}},
                    {"graphs.masm_cbor_graph": {"$eq": product_graphs}},
                    {"label": {"$in": ["minimum_optimized", "user_optimized"]}},
                    {"compound": {"$ne": ""}},
                ]
            }

            selections.append(json.dumps(selection))
            mol_files.append(mol_file)

        return selections, mol_files

    def _get_energy(self, structure: db.Structure, prop_name: str) -> Union[float, None]:
        """
        Gives energy value depending on demanded property. If the property does not exit, None is returned.
        Adapted from the kinetics gear.

        Parameters
        ----------
        structure : scine_database.Structure (Scine::Database::Structure)
            The structure we want the energy from
        prop_name : str
            The name of the energy property such as 'electronic_energy' or 'gibbs_free_energy'

        Returns
        -------
        Union[float, None]
            energy value in Hartree
        """
        structure.link(self._structures)

        if not structure.has_property(prop_name):
            return None
        prop = db.NumberProperty(structure.get_property(prop_name))
        prop.link(self._properties)

        return prop.get_data()

    def _get_first_structure_of_compound(self, compound: db.Compound):
        """
        Obtain the structure of an unlinked compound. The structure will be automatically linked to
        the database.

        Parameters
        ----------
        compound :: db.Compound
            Compound of interest

        Returns
        -------
        first_structure :: db.Structure
            The first structure of a compound, linked to the database.
        """

        compound.link(self._compounds)
        first_structure_id = compound.get_centroid()
        first_structure = self._structures.get_structure(first_structure_id)
        first_structure.link(self._structures)

        return first_structure

    def _get_frequencies(self, structure: db.Structure) -> Union[np.ndarray, None]:
        """
        Get the frequencies of a structure in cm^-1.

        Parameters
        ----------
        compound :: db.Structure
            Structure of interest, already linked to collection

        Returns
        -------
        frequencies :: numpy.ndarray
            The frequencies of the given structrue in cm^-1 as 1D array.
        """
        if len(structure.get_atoms().elements) < 2:
            return None
        # # # Get frequencies
        prop_id = structure.get_property("frequencies")
        prop = self._properties.get_vector_property(prop_id)
        prop.link(self._properties)
        frequencies = prop.get_data() * utils.INVERSE_CENTIMETER_PER_HARTREE * utils.PI * 2

        return frequencies

    def _setup_plt(self):
        """
        Setup matplotlib.plt

        """
        plt.style.use("seaborn-whitegrid")
        plt.rc("font", size=12)
        plt.rc("axes", labelsize=16, titlesize=8)
        #     plt.rc('title', labelsize=16)
        plt.grid(True, color="black", lw=0.6, alpha=0.4, linestyle="--", dashes=(2, 1))

        plt.close()

    def _setup_ax(self, ax):
        """
        Formats an ax object to PLT's standard format.

        Parameter
        ----------
            ax :: plt.axes.Axes
                The axes object to be modified.

        Return
        ----------
            ax :: plt.axes.Axes
                The modified axes object.
        """
        # # # Style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_color("black")
        ax.spines["left"].set_linewidth(1.5)
        ax.tick_params(direction="out", length=5, width=1.5, colors="black")
        #     ax.tick_params(axis='x',labelsize=14)
        #     ax.tick_params(axis='y',labelsize=12)

        return ax
