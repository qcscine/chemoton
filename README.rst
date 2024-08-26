.. image:: docs/source/res/chemoton_header.png
   :alt: SCINE Chemoton

.. inclusion-marker-do-not-remove

Introduction
------------

With Chemoton you can explore complex chemical reaction networks in an automated
fashion. Based on a Python framework, workflows can be built that probe reactivity
of chemical systems with quantum chemical methods. Various quantum chemical software
programs and job schedulers are supported by the back-end software SCINE Puffin.

License and Copyright Information
---------------------------------

Chemoton is distributed under the BSD 3-clause "New" or "Revised" License.
For more license and copyright information, see the file ``LICENSE.txt`` in the
repository.

Installation
------------

Prerequisites
.............

The key requirements for Chemoton are the Python packages ``scine_utilities``,
``scine_database``, and ``scine_molassember``. These packages are available from
PyPI and can be installed using ``pip``.
However, these packages can also be compiled from sources. For the latter case please
visit the repositories of each of the packages and follow their guidelines or
bootstrap a `puffin <https://github.com/qcscine/puffin>`_ which will install the same
dependencies.

Installation
............

Chemoton can be installed using ``pip`` (``pip3``) once the repository has been cloned:

.. code-block:: bash

   git clone https://github.com/qcscine/chemoton.git
   cd chemoton
   pip install -r requirements.txt
   pip install .

A non-root user can install the package using a virtual environment, or
the ``--user`` flag.

The documentation can be found online, or it can be built using:

.. code-block:: bash

   make -C docs html

It is then available at:

.. code-block:: bash

   <browser-name> docs/build/html/index.html

In order to build the documentation, you need a few extra Python packages, which
are not installed automatically together with Chemoton. In order to install them,
run

.. code-block:: bash

   pip install -r requirements-dev.txt

Tutorial
--------

Minimal Example
...............

Assuming that Chemoton has successfully been installed, a small example
exploration can be started by running Chemoton's main function.
It requires a database running on ``localhost`` listening to the default
``mongodb`` port ``27017``; additionally a ``puffin`` instance has to be
running and checking the database named ``default``.

Setting up these things may look somewhat like this:

1. Start a ``mongodb`` server. Limit its memory usage and maybe customize where
to log and store the data.

.. code-block:: bash

   mongod --fork --port=27017 -dbpath=<path to db storage dir> -wiredTigerCacheSizeGB=1 --logpath=mongo.log

2. Configure and bootstrap a ``puffin``:

.. code-block:: bash

   pip install scine-puffin
   python3 -m scine_puffin configure
   # Edit the generated puffin.yaml here
   python3 -m scine_puffin -c puffin.yaml bootstrap

3. Source the ``puffin`` settings and tell it to listen to the correct DB.
(Hostname and port should be the default ones.) Then start it.

.. code-block:: bash

   source puffin.sh
   export PUFFIN_DATABASE_NAME=default
   python3 -m scine_puffin -c puffin.yaml start

4. Run the Chemoton exploration defined in the ``__main__`` function:

.. code-block:: bash

   python3 -m scine_chemoton wipe

The optional ``wipe`` argument will start the example exploration with a clean
``default`` DB; giving the ``continue`` argument will reuse old data.

Expanding on the Minimal Example
................................

The functionalities used in Chemoton's ``__main__.py`` are a good starting point
for most simple explorations. The file contains a lot of settings that are
explicitly set to their defaults in order to show their existence.

While we recommend to read the documentation of Chemoton, tinkering with
explorations can be as simple as:

.. code-block:: bash

   chemoton_main=$(python3 -c 'from scine_chemoton import __main__ as m; print(m.__file__)')
   echo $chemoton_main
   cp $chemoton_main my_awesome_exploration.py

and editing the file to your liking: disabling gears, adding filters or
just changing methods.

In order to directly have analysis tools for the network at hand
or run explorations without coding,
we recommend our graphical user interface 
`Heron <https://github.com/qcscine/heron>`_.


How to Cite
-----------

When publishing results obtained with Chemoton, please cite the corresponding
release as archived on `Zenodo <https://doi.org/10.5281/zenodo.6695583>`_ (DOI
10.5281/zenodo.6695583; please use the DOI of the respective release).

In addition, we kindly request you to cite the following article when using Chemoton:

J. P. Unsleber, S. A. Grimmel, M. Reiher,
"Chemoton 2.0: Autonomous Exploration of Chemical Reaction Networks",
*J. Chem. Theory Comput.*, **2022**, *18*, 5393.

If you are applying SCINE Pathfinder in your exploration or analysis, we kindly request you to cite the following article:

P. L. Türtscher, M. Reiher,
"Pathfinder - Navigating and Analyzing Chemical Reaction Networks with an Efficient Graph-Based Approach",
*J. Chem. Inf. Model.*, **2023**, *63*, 147.

If you are applying kinetic modeling in your exploration or analysis, we kindly request you to cite the following article:
J. Proppe, M. Reiher,
"Mechanism Deduction from Noisy Chemical Reaction Networks",
*J. Chem. Theory Comput.*, **2019**, *15*, 357.

M. Bensberg, M. Reiher,
"Concentration-Flux-Steered Mechanism Exploration with an Organocatalysis Application",
*Isr. J. Chem.*, **2023**, *63*, 147.

If you are applying the Steering Wheel in your exploration, we kindly request you to cite the following article:
M. Steiner, M. Reiher,
"A human-machine interface for automatic exploration of chemical reaction networks",
*Nat. Commun.*, **2024**, 15, 3680.

Furthermore, when publishing results obtained with any SCINE module, please cite the following paper:

T. Weymuth, J. P. Unsleber, P. L. Türtscher, M. Steiner, J.-G. Sobez, C. H. Müller, M. Mörchen,
V. Klasovita, S. A. Grimmel, M. Eckhoff, K.-S. Csizi, F. Bosia, M. Bensberg, M. Reiher,
"SCINE—Software for chemical interaction networks", *J. Chem. Phys.*, **2024**, *160*, 222501
(DOI `10.1063/5.0206974 <https://doi.org/10.1063/5.0206974>`_).

Support and Contact
-------------------

In case you should encounter problems or bugs, please write a short message
to scine@phys.chem.ethz.ch.
