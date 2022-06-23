Gears
=====

Interface
---------
.. autoclass:: scine_chemoton.gears.Gear
    :special-members: __call__

Implementations
---------------

Compound Sorting
~~~~~~~~~~~~~~~~
.. automodule:: scine_chemoton.gears.compound

Conformer Generation
~~~~~~~~~~~~~~~~~~~~
.. automodule:: scine_chemoton.gears.conformers.brute_force

Elementary Steps Exploration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: scine_chemoton.gears.elementary_steps.ElementaryStepGear
.. automodule:: scine_chemoton.gears.elementary_steps.brute_force
.. automodule:: scine_chemoton.gears.elementary_steps.minimal
.. autoclass:: scine_chemoton.gears.elementary_steps.trial_generator.TrialGenerator
.. automodule:: scine_chemoton.gears.elementary_steps.trial_generator.bond_based
.. automodule:: scine_chemoton.gears.elementary_steps.trial_generator.fragment_based
.. automodule:: scine_chemoton.gears.elementary_steps.trial_generator.connectivity_analyzer

Compound Filter
"""""""""""""""
.. automodule:: scine_chemoton.gears.elementary_steps.compound_filters

Reactive Site Filter
""""""""""""""""""""
.. automodule:: scine_chemoton.gears.elementary_steps.reactive_site_filters

Kinetics
~~~~~~~~
.. automodule:: scine_chemoton.gears.kinetics

Reaction Sorting
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: scine_chemoton.gears.reaction

Network Refinement
~~~~~~~~~~~~~~~~~~
.. automodule:: scine_chemoton.gears.refinement

Job Scheduling
~~~~~~~~~~~~~~
.. automodule:: scine_chemoton.gears.scheduler

Thermodynamic Data Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: scine_chemoton.gears.thermo
