Changelog
=========

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
