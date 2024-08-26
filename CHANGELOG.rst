Changelog
=========

Release 4.0.0
-------------

API Changes:
 - Move network refinement to separate folder
 - Move all filters to separate folder
 - Move reaction rules to separate folder
 - Remove deprecated database queries
 - Removed some gear options such as `pre_refine_model` in favor of the default `model` field
 - Set the default `model` option with `PlaceHolderModel` and check if it is still present before launching the gear to make sure that an intended model is used
 - Small changes to the default arguments of the kinetic modeling gear to facilitate its use in SCINE Heron

New features
 - Add StructureFilters derived from fitting AggregatesFilters.
 - Assemble reactive complexes from two given molecules easily with an CLI.
 - Elementary Step Gears can set up reaction trials for flasks.
 - Almost all Aggregate and ReactiveSiteFilters are made suitable for flasks. If not, a note is included in the documentation.
 - Add an AggregateFilter that filters based on substructure searches.
 - Add an AggregateFilter that filters based on allowed / disallowed molecular charges.
 - Add reverse option for some existing AggregateFilters.
 - KineticsGear now has an option that allows to run it as many cycles as long as no new aggregates are activated and then stops itself. 
 - Add gear to carry out additional energy calculations.
 - Kinetic modeling may now work with a mixture of electronic structure models.
 - Energies may now be automatically referenced as formation energies from the atoms.
 - The bond-based trial generator is now able to enable (enable exploration/analysis) the analysis of structures,
   elementary steps, etc. in the results of already completed calculations, i.e., the trial generator detects that
   it already set up a calculation previously. Instead of just continuing, it may then enable the results of this
   calculation. This simulates a step-wise exploration of already existing data in the DB that was disabled before
   and is intended to allow re-exploration with different reaction conditions such as temperature and pressure or
   recycle existing data after changing the kinetics gear (or kinetic modeling, pathfinder etc.).
 - Add a framework to filter reactions (analogous to the aggregates filters) and filter elementary steps. These
   new filters can be applied to the refinement framework (e.g., barrier-screening, barrier-less reaction selection,
   concentrations etc.).
 - The refinement is now split into 3 gears:
    - Calculation-based refinement looping over the calculations and refining elementary steps (structures etc).
    - Reaction-based refinement looping over reactions and selects elementary steps to be refined for each reaction
      (including energy window for the step selection).
    - Aggregate-based refinement looping over aggregates and refining the structures of these aggregates (including
      energy window for the selection of the structure).
 - All refinement gears support the same enabling strategy as introduced to the bond-based trial generator.
 - Add reaction filter-based kinetics gear.
 - Add reaction filter to constrain the maximum energy encountered when exploring a single potential energy surface.
 - Add an aggregate filter to enforce that the particle number is conserved during the exploration.
 - Add feature that a running Network Expansion of the Steering Wheel can be interrupted and continued later.

Changes
 - Rework the Engine / Gear interaction, by replacing the sent signals with shared memory members
 - Add EngineHandler class to join any forked engines if a signal is sent, replacing the custom code in the main script; the class also allows for running all engines and stopping the processes gracefully
 - Change the internal representation of some AggregateFilters from strings to enums in order to be faster.
 - Add additional caches to some AggregateFilters for increased performance.
 - Add more default settings.

Bugfixes
 - The MinimalConnectivityKinetics and BasicBarrierHeightKinetics did not consider reactions where all reactants on the right hand side were available, which lead to fewer activated aggregates
 - GearOptions of the NetworkExpansion did not consider that there could be multiple gears of the same type but with different options in a protocol, hence their datastructure (including access keys) are now changed.
 - Fix type annotations in the documentation.

Technical changes
 - `get_transition_state_free_energy` in the reaction_wrapper now returns `max(e_lhs, e_rhs)` if the reaction is
   barrier-less.
 - Add more typehints

Release 3.1.0
-------------

New features
 - Add SteeringWheel infrastructure for actively steering explorations.
 - Add ThermoAggregateHousekeeping gear which allows sorting of structures with a frequency check.
 - AtomPairFunctionalGroupFilter for specifying pairs of functional groups that are allowed to react.
 - CentralSiteFilter to focus explorations on certain elements, suited well for homogeneous catalysis.
 - New ElementaryStepGear to focus explorations on certain structures.
 - New ReactiveSiteFilter based on substructures provided in .xyz or .mol files. 
 - The KineticModelingGear is now able to setup jobs for the puffin interface of the ReactionMechanismSimulator.

Changes
 - Queries and utility functions related only to the database are moved to the `scine_database` package. The functionality here is deprecated and the unittests are removed.
 - Update address in license

Release 3.0.0
-------------

New features
 - Add improved handling of kill (SIGINT) and terminate (SIGTERM) signals to engines,
   including new breakpoints in existing gears.
 - Add a gear that allows the re-running of calculations that failed (e.g., failure to locate a transition state).
 - Allow to either exclude or include reactive sites based on rules. 
 - New PathfinderKinetics gear to activate compounds based on the compound costs obtained with Pathfinder.
 - New TrialGenerator for ElementaryStepGears based on reaction templates.

Improvements
 - Many gears and also filters now use local caches for enhanced performance.
 - All gears have an Options object holding at least a Model.
 - All ElementaryStepGears and TrialGenerators expose a public method that allows access to the calculations
   that would be set-up in the next run.
 - Add option to ElementaryStepGear and TrialGenerator that allows to check all existing calculations for
   an exact settings match, so that elementary step trials can be enhanced with more inclusive options.
 - Allow to get all valid compounds for the BruteForceConformersGear
 - Add caches to ElementaryStepGear and BasicBarrierHeightKinetics
 - More gears can be limited with an AggregateFilter.
 - Add type checking of reaction rules at runtime.
 - More options to chose for building a graph with Pathfinder and more robust determination of compound costs.
 - Allow restriction of compounds based on maximum reaction energy of reactions leading to them.

Changes
 - Separate the reaction rule definitions from the reactive site filters and structure them.
 - Redefine the FunctionalGroupRule.
 - Rename CompoundFilter to AggregateFilter.
 - Consider the explore status of each aggregate/reaction for the Thermo gear and add setting to allow to ignore
   the status.
 - Increase default number of optimization cycles for reactive complex optimization to find a potential
   barrierless elementary step.

Bug Fixes:
 - Add the calculation status to the safety query of the AggregateHousekeeping gear if the found structure is
   the result of a minimization to avoid false positives due to race conditions with the results-adding puffin.
 - Fix lastmodified query to correctly handle time zones.
 - Fix bug in attack direction cache of the reactive complex generator.
 - Fix bug in BasicBarrierHeightKinetics leading to too many activations in certain network arrangements.

Release 2.2.0
-------------

New features
 - Introduce Pathfinder, a graph-based approach to analyze how compounds are connected via reactions while considering
   kinetic and stoichiometric constraints.

Release 2.1.0
-------------

New features
 - Introduce Flasks to the reaction networks (aggregates of stable non-bonded complexes)
 - Elementary-step gear that uses the current minimum-energy conformer for reaction trial generation.
 - Added a gear that sets up kinetic modeling jobs.
 - Allow the refinement of a subset of elementary steps per reaction. The subset is given through an energy cut-off
   above the lowest lying transition state.
 - Introduce possibility to efficiently explore barrierless dissociations.

Release 2.0.0
-------------

Python rewrite, and open source release with the following initial features:
 - Scriptable framework including a base set of features for the automated
   exploration of chemical reaction networks
 - Initial chemical reaction networks consisting of
    - Structures aggregated into Compounds
    - Elementary Steps aggregated into Reactions
    - Properties tagged to Structures
    - Calculations that generated the network
 - Definitions of ``Engines`` with perpetually running ``Gears`` to continuously
   perform tasks with chemical reaction networks (see list below)
 - Storage and expansion of chemical reaction networks in a SCINE Database
 - Automated job set up and execution via SCINE Puffin
 - Definitions of basic filters to reduce number of Elementary Step trials
   (see list below)

Initial ``Engines``/``Gears``:
 - Basic bookkeeping jobs:
    - Sorting Structures into Compounds (BasicCompoundHousekeeping)
    - Sorting Elementary Steps into Reactions (BasicReactionHousekeeping)
    - Basic Scheduling and prioritization of Calculations (Scheduler)
 - Data completion jobs:
    - Conformer generation per compound (BruteForceConformers)
    - Hessian generation per transition state and minimum energy Structure
      (BasicThermoDataCompletion)
 - Elementary Step exploration based on existing Compounds:
    - For one Structure per Compound (MinimalElementarySteps):
       - Based on atoms/fragments (AFIR, NT1)
       - Based on bonds (NT2)
    - For all combinations of Structures per Compounds (BruteForceElementarySteps):
       - Based on atoms/fragments (AFIR, NT1)
       - Based on bonds (NT2)
 - Steering of network growth via simple kinetic analyses:
    - Based on connectivity to user input (MinimalConnectivityKinetics)
    - Based on barrier heights of Elementary Steps (BasicBarrierHeightKinetics)

Initial set of filters:
  - Compound filtering possible:
     - Base class, allows all compounds (CompoundFilter)
     - By element counts (ElementCountFilter, ElementSumCountFilter)
     - By atom counts or molecular weights (MolecularWeightFilter, AtomNumberFilter)
     - By database IDs (IDFilter, OneCompoundIDFilter, SelectedCompoundIDFilter)
     - By context (SelfReactionFilter)
     - By Hessian evaluation (TrueMinimumFilter)
     - By composition (CatalystFilter)
  - Reactive site filtering possible:
     - Base class, allows all reactive sites (ReactiveSiteFilter)
     - By fixed, simple rankings (SimpleRankingFilter, MasmChemicalRankingFilter)
     - By custom user rules (AtomRuleBasedFilter, FunctionalGroupRule)
     - By atom types (ElementWiseReactionCoordinateFilter)
  - All filters of the same type can be chained with logical operations to
    tailor the behaviour

Release 1.0.0
-------------

Closed source C++ prototype implementation.
