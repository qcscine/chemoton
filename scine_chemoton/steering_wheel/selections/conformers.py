#!/usr/bin/env python3
from __future__ import annotations
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import numpy as np
import scine_database as db
from scine_database.energy_query_functions import get_energy_for_structure
from scine_database.queries import optimized_labels_enums
import scine_utilities as utils

from . import Selection
from ..datastructures import SelectionResult, LogicCoupling, NetworkExpansionResult
from scine_chemoton.filters.aggregate_filters import AggregateFilter, SelectedAggregateIdFilter
from scine_chemoton.filters.reactive_site_filters import ReactiveSiteFilter


class ConformerSelection(Selection, ABC):
    """
    The base class for all conformer selections.
    This class is not meant to be used directly, but rather as a base class for other conformer selections.
    """

    class Options(Selection.Options):

        def __init__(self, model: db.Model, include_thermochemistry: bool, *args, **kwargs):
            super().__init__(model, *args, **kwargs)
            self.include_thermochemistry = include_thermochemistry

    options: ConformerSelection.Options  # required for mypy checks, so it knows which options object to check

    def _have_aggregates_sanity_check(self) -> NetworkExpansionResult:
        """
        Ensures that the given step result has aggregates.
        """
        step_result = self.get_step_result()
        if not step_result.compounds and not step_result.flasks:
            raise RuntimeError(f"{self.name} received step result {step_result}, which is missing aggregates.")
        return step_result

    def _get_allowed_structures(self, aggregate: Union[db.Compound, db.Flask]) -> List[db.Structure]:
        allowed_labels = optimized_labels_enums()
        structures = [s for s in aggregate.get_structures(self._manager)
                      if s.get_label() in allowed_labels and s.get_model() == self.options.model]
        if not structures:
            raise RuntimeError(f"Aggregate {aggregate.id()} is missing structures with model {self.options.model}.")
        return structures

    def _select(self) -> SelectionResult:
        """
        Handles base selection logic, and calls the _structure_selection method to select the structures from
        each aggregate.
        """
        step_result = self._have_aggregates_sanity_check()

        relevant_aggregate_ids = []
        relevant_structure_ids = []

        for i, aggregates in enumerate([step_result.compounds, step_result.flasks]):
            for agg_id in aggregates:
                aggregate: Union[db.Compound, db.Flask] = \
                    db.Compound(agg_id, self._compounds) if i == 0 else db.Flask(agg_id, self._flasks)
                relevant_aggregate_ids.append(str(agg_id))
                relevant_structure_ids += self._structure_selection(aggregate)
        return SelectionResult(aggregate_filter=SelectedAggregateIdFilter(relevant_aggregate_ids),
                               structures=relevant_structure_ids)

    @abstractmethod
    def _structure_selection(self, aggregate: Union[db.Compound, db.Flask]) -> List[db.ID]:
        """
        Method to be implemented by subclasses to select the structures from the given aggregate.

        Parameters
        ----------
        aggregate : Union[db.Compound, db.Flask]
            The aggregate to select structures from.

        Returns
        -------
        List[db.ID]
            The list of structure IDs to be selected.
        """


class CentroidConformerSelection(ConformerSelection):
    """
    Simply selects the centroid of each aggregate, which is the first structure that was found for that aggregate.
    """

    class Options(ConformerSelection.Options):
        def __init__(self, model: db.Model, *args, **kwargs):
            # implemented to make sure that the include_thermochemistry option is set to False
            include_thermochemistry = False
            super().__init__(model, include_thermochemistry, *args, **kwargs)

    options: CentroidConformerSelection.Options  # required for mypy checks, so it knows which options object to check

    def _structure_selection(self, aggregate: Union[db.Compound, db.Flask]) -> List[db.ID]:
        structures = self._get_allowed_structures(aggregate)
        return [structures[0].id()]


class LowestEnergyConformerSelection(ConformerSelection):
    """
    Selects the lowest energy conformer of each aggregate.
    """

    options: LowestEnergyConformerSelection.Options

    def __init__(self, model: db.Model,  # pylint: disable=keyword-arg-before-vararg
                 additional_aggregate_filters: Optional[List[AggregateFilter]] = None,
                 additional_reactive_site_filters: Optional[List[ReactiveSiteFilter]] = None,
                 logic_coupling: Union[str, LogicCoupling] = LogicCoupling.AND,
                 include_thermochemistry: bool = False, *args, **kwargs):
        super().__init__(model, additional_aggregate_filters, additional_reactive_site_filters, logic_coupling,
                         include_thermochemistry, *args, **kwargs)
        self.energy_type = "gibbs_free_energy" if self.options.include_thermochemistry else "electronic_energy"

    def _structure_selection(self, aggregate: Union[db.Compound, db.Flask]) -> List[db.ID]:
        structures = self._get_allowed_structures(aggregate)
        lowest_energy = None
        lowest_energy_sid = None
        for structure in structures:
            energy = get_energy_for_structure(structure, self.energy_type, self.options.model,
                                              self._structures, self._properties)
            if energy is None:
                raise RuntimeError(f"Missing energy of type {self.energy_type} "
                                   f"for structure {structure.id()} of aggregate {aggregate.id()}.")
            if lowest_energy is not None and energy > lowest_energy:
                structure.disable_exploration()
            else:
                if lowest_energy_sid is not None:
                    previous_structure = db.Structure(lowest_energy_sid, self._structures)
                    previous_structure.disable_exploration()
                lowest_energy = energy
                lowest_energy_sid = structure.id()
        if lowest_energy_sid is None:
            raise RuntimeError("Could not determine a lowest energy conformer")
        return [lowest_energy_sid]


class ClusterSelection(ConformerSelection, ABC):
    """
    A base class for all conformer selections that rely on clustering.
    This class is not meant to be used directly, but rather as a base class for other conformer selections.
    """

    class Options(ConformerSelection.Options):

        def __init__(self, model: db.Model, include_thermochemistry: bool,
                     n_clusters: Optional[int], cluster_rmsd_cutoff: Optional[float], *args, **kwargs):
            super().__init__(model, include_thermochemistry, *args, **kwargs)
            self.n_clusters = n_clusters
            self.cluster_rmsd_cutoff = cluster_rmsd_cutoff

    options: ClusterSelection.Options  # required for mypy checks, so it knows which options object to check

    def __init__(self, model: db.Model,
                 additional_aggregate_filters: Optional[List[AggregateFilter]] = None,
                 additional_reactive_site_filters: Optional[List[ReactiveSiteFilter]] = None,
                 logic_coupling: Union[str, LogicCoupling] = LogicCoupling.AND,
                 include_thermochemistry: bool = False,
                 n_clusters: Optional[int] = None,
                 cluster_rmsd_cutoff: Optional[float] = None):
        super().__init__(model, additional_aggregate_filters, additional_reactive_site_filters, logic_coupling,
                         include_thermochemistry, n_clusters, cluster_rmsd_cutoff)
        if self._n_clusters_selected() and self._rmsd_selected():
            raise ValueError("Cannot specify number of clusters and cluster distance cutoff simultaneously, only one")
        if not self._n_clusters_selected() and not self._rmsd_selected():
            if self.options.n_clusters is not None and self.options.n_clusters < 0:
                raise ValueError("Number of clusters must be positive")
            if self.options.cluster_rmsd_cutoff is not None and self.options.cluster_rmsd_cutoff < 0:
                raise ValueError("Cluster distance cutoff must be positive")
            raise ValueError("Must specify either number of clusters or cluster distance cutoff")
        self.energy_type = "gibbs_free_energy" if self.options.include_thermochemistry else "electronic_energy"

    def _n_clusters_selected(self) -> bool:
        return self.options.n_clusters is not None and self.options.n_clusters > 0

    def _rmsd_selected(self) -> bool:
        return self.options.cluster_rmsd_cutoff is not None and self.options.cluster_rmsd_cutoff > 1e-6

    @staticmethod
    def _distance_matrix(structures: List[db.Structure]) -> np.ndarray:
        """
        Constructs a nxn matrix of distances between all pairs of n structures.

        Parameters
        ----------
        structures : List[db.Structure]
            The list of structures to calculate the distance matrix for.

        Returns
        -------
        np.ndarray
            The distance matrix.
        """
        n = len(structures)
        list_of_positions = [structure.get_atoms().positions for structure in structures]
        matrix = np.zeros(shape=(n, n))
        for i in range(n):
            ref = list_of_positions[i]
            for j in range(i + 1, n):
                pos = list_of_positions[j]
                fit = utils.QuaternionFit(ref, pos)
                value = fit.get_rmsd()
                matrix[i][j] = value
                matrix[j][i] = value
        return matrix

    def _cluster_assignments(self, distance_matrix: np.ndarray) -> List[List[int]]:
        linkage = hierarchy.linkage(squareform(distance_matrix))
        if self._n_clusters_selected():
            clustering_result = hierarchy.fcluster(linkage, t=self.options.n_clusters, criterion="maxclust")
        elif self._rmsd_selected():
            clustering_result = hierarchy.fcluster(linkage, t=self.options.cluster_rmsd_cutoff, criterion="distance")
        else:
            raise RuntimeError
        cluster_assignments = clustering_result - 1
        clusters: List[List[int]] = [[] for _ in range(np.max(cluster_assignments) + 1)]
        for i, assignment in enumerate(cluster_assignments):
            clusters[assignment].append(i)
        return clusters

    def _structure_selection(self, aggregate: Union[db.Compound, db.Flask]) -> List[db.ID]:
        structures = self._get_allowed_structures(aggregate)
        if len(structures) == 1:
            return [structures[0].id()]
        distance_matrix = self._distance_matrix(structures)
        cluster_selection = self._cluster_selection(structures, distance_matrix)
        return [structure.id() for i, structure in enumerate(structures) if i in cluster_selection]

    @abstractmethod
    def _cluster_selection(self, structures: List[db.Structure], distance_matrix: np.ndarray) -> List[int]:
        pass


class ClusterCentroidConformerSelection(ClusterSelection):

    options: ClusterCentroidConformerSelection.Options

    def _cluster_selection(self, structures: List[db.Structure], distance_matrix: np.ndarray) -> List[int]:

        clusters = self._cluster_assignments(distance_matrix)
        centroids = []
        for cluster in clusters:
            min_rmsd = None
            min_entry = None
            for entry in cluster:
                val = sum(distance_matrix[entry][i] for i in cluster)
                if min_rmsd is None or val < min_rmsd:
                    min_rmsd = val
                    min_entry = entry
            if min_entry is None:
                raise RuntimeError
            centroids.append(min_entry)
        return centroids


class LowestEnergyConformerPerClusterSelection(ClusterSelection):

    options: LowestEnergyConformerPerClusterSelection.Options

    def _cluster_selection(self, structures: List[db.Structure], distance_matrix: np.ndarray) -> List[int]:

        clusters = self._cluster_assignments(distance_matrix)
        sele = []
        for cluster in clusters:
            min_energy = None
            min_entry = None
            for entry in cluster:
                val = get_energy_for_structure(structures[entry], self.energy_type, self.options.model,
                                               self._structures, self._properties)
                if min_energy is None or (val is not None and val < min_energy):
                    min_energy = val
                    min_entry = entry
            if min_entry is None:
                raise RuntimeError
            sele.append(min_entry)

        return sele
