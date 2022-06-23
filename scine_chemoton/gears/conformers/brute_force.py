#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import time
from json import dumps

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application imports
from .. import Gear
from ...utilities.queries import model_query, stop_on_timeout


class BruteForceConformers(Gear):
    """
    This Gear generates all possible conformers as a guess based on the centroid
    of each Compound. The guesses are then optimized. Deduplication of optimized
    Structures happens when sorting them into compounds.

    Attributes
    ----------
    options :: BruteForceConformers.Options
        The options for the BruteForceConformers Gear.

    Notes
    -----
    Checks if any existing Compound does not have conformer guesses generated.
    Conformer guesses are supposed to be the only 'minimum_guess' structures
    assigned to compounds as they are the only ones where the correct graph
    assignment should be possible.
    If no confromer guesses are present they are generated and the associated
    geometry optimizations are scheduled.
    """

    class Options:
        """
        The options for the BruteForceConformers Gear.
        """

        __slots__ = (
            "cycle_time",
            "model",
            "conformer_job",
            "minimization_job",
            "conformer_settings",
            "minimization_settings",
        )

        def __init__(self):
            self.cycle_time = 10
            """
            int
                The minimum number of seconds between two cycles of the Gear.
                Cycles are finished independent of this option, thus if a cycle
                takes longer than the cycle_time will effectively lead to longer
                cycle times and not cause multiple cycles of the same Gear.
            """
            self.model: db.Model = db.Model("PM6", "", "")
            """
            db.Model (Scine::Database::Model)
                The Model used for the conformer generation.
                The default is: PM6 using Sparrow.
            """
            self.conformer_job: db.Job = db.Job("conformers")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used for the generation of new conformer guesses.
                The default is: the 'conformers' order on a single core.
            """
            self.conformer_settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
                Additional settings passed to the conformer generation
                calculations. Empty by default.
            """
            self.minimization_job: db.Job = db.Job("scine_geometry_optimization")
            """
            db.Job (Scine::Database::Calculation::Job)
                The Job used to optimize the geometries of the generated conformer
                guesses.
                The default is: the 'scine_geometry_optimization' order on a single
                core.
            """
            self.minimization_settings: utils.ValueCollection = utils.ValueCollection()
            """
            utils.ValueCollection
                Additional settings passed to the geometry optimization
                calculations. Empty by default.
            """

    def __init__(self):
        super().__init__()
        self.options = self.Options()
        self._calculations = "required"
        self._structures = "required"
        self._compounds = "required"
        # local cache variables
        self._completed = []

    def _loop_impl(self):
        # Loop over all compounds
        for compound in stop_on_timeout(self._compounds.iterate_all_compounds()):
            compound.link(self._compounds)

            # Check for initial reasons to skip
            if not compound.explore():
                continue
            if compound.id().string() in self._completed:
                continue

            # ============================== #
            #  Conformer Guess Generation    #
            # ============================== #
            has_guesses = False
            selection = {
                "$and": [
                    {"job.order": self.options.conformer_job.order},
                    {"auxiliaries": {"compound": {"$oid": compound.id().string()}}},
                ]
                + model_query(self.options.model)
            }
            hit = self._calculations.get_one_calculation(dumps(selection))
            if hit is not None:
                conformer_generation = hit
                conformer_generation.link(self._calculations)
                # If the job is done the are guesses (maybe)
                if conformer_generation.get_status() in [
                    db.Status.COMPLETE,
                    db.Status.ANALYZED,
                ]:
                    has_guesses = True
            else:
                # Generate a conformer job if there was none
                centroid = db.Structure(compound.get_centroid())
                centroid.link(self._structures)
                # Setup conformer generation
                conformer_generation = db.Calculation()
                conformer_generation.link(self._calculations)
                conformer_generation.create(self.options.model, self.options.conformer_job, [centroid.id()])
                conformer_generation.set_auxiliary("compound", compound.id())
                if self.options.conformer_settings:
                    conformer_generation.set_settings(self.options.conformer_settings)
                conformer_generation.set_status(db.Status.HOLD)
                # Continue as there can not be any guesses yet
                continue

            # ====================== #
            #   Guess Optimization   #
            # ====================== #
            optimized_structures = 0
            if has_guesses:
                results = conformer_generation.get_results()
                conformer_guesses = results.get_structures()
                if len(conformer_guesses) == 0:
                    # if there are no guesses then this compound is done
                    self._completed.append(compound.id())
                    continue
                for guess in conformer_guesses:
                    model = self.options.model
                    selection = {
                        "$and": [
                            {"job.order": self.options.minimization_job.order},
                            {"structures": {"$oid": guess.string()}},
                        ]
                        + model_query(model)
                    }
                    hit = self._calculations.get_one_calculation(dumps(selection))
                    if hit is None:
                        # Generate them if they are not
                        minimization = db.Calculation()
                        minimization.link(self._calculations)
                        minimization.create(model, self.options.minimization_job, [guess])
                        # Sleep a bit in order not to make the DB choke
                        time.sleep(0.001)
                        if self.options.minimization_settings:
                            minimization.set_settings(self.options.minimization_settings)
                        minimization.set_status(db.Status.HOLD)
                        time.sleep(0.001)
                    else:
                        minimization = hit
                        minimization.link(self._calculations)
                        if minimization.get_status() == db.Status.FAILED:
                            # Failed optimizations shall not stop the completion
                            optimized_structures += 1
                        elif minimization.get_status() not in [
                            db.Status.COMPLETE,
                            db.Status.ANALYZED,
                        ]:
                            pass
                        else:
                            optimized_structures += 1

                # If all optimizations are done mark this compound as complete
                if optimized_structures == len(conformer_guesses):
                    self._completed.append(compound.id())
