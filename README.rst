.. image:: docs/source/res/chemoton_header.png
   :alt: SCINE Chemoton

.. inclusion-marker-do-not-remove

Introduction
------------

With Chemoton you can explore complex chemical reaction networks in an automated
fashion. Based on a Python framework, workflows can be built that probe reactivity
of chemical systems with quantum chemical methods. Various quantum chemical software
programs and job schedulers are supported via the back-end software SCINE Puffin.

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

   cp <chemoton-git>/scine_chemoton/__main__.py my_awesome_exploration.py

and editing the file to your liking: disabling gears, adding filters or
just changing methods.

How to Cite
-----------

When publishing results obtained with Chemoton, please cite the corresponding
release as archived on `Zenodo <https://doi.org/10.5281/zenodo.6695583>`_ (DOI
10.5281/zenodo.6695583; please use the DOI of the respective release).

In addition, we kindly request you to cite the following article when using Chemoton:

J. P. Unsleber, S. A. Grimmel, M. Reiher,
"Chemoton 2.0: Autonomous Exploration of Chemical Reaction Networks",
*J. Chem. Theory Comput.*, **2022**, *18*, 5393.

Support and Contact
-------------------

In case you should encounter problems or bugs, please write a short message
to scine@phys.chem.ethz.ch.
