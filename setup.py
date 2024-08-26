from os import path
from setuptools import setup, find_packages
import sys


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 6)
if sys.version_info < min_version:
    error = """
Chemoton does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(
        *(sys.version_info[:2] + min_version)
    )
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.rst"), encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open(path.join(here, "requirements.txt")) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines() if not line.startswith("#")]

with open(path.join(here, 'scine_chemoton', '_version.py')) as f:
    exec(f.read())


setup(
    name="scine_chemoton",
    version=__version__,
    description="Software driving the automated exploration of chemical reaction networks",
    long_description=readme,
    author="ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group",
    author_email="scine@phys.chem.ethz.ch",
    url="https://www.scine.ethz.ch",
    python_requires=">={}".format(".".join(str(n) for n in min_version)),
    packages=find_packages(include=["scine_chemoton", "scine_chemoton.*"],
                           exclude=["scine_chemoton.tests*"]),
    entry_points={
        "console_scripts": [
            'scine_chemoton_create_reactive_complex = scine_chemoton.__init__:create_reactive_complex_cli',
            # 'command = some.module:some_function',
        ],
    },
    include_package_data=True,
    package_data={
        "scine_chemoton": [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
            "resources/*xyz"
        ]
    },
    install_requires=requirements,
    license="BSD (3-clause)",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: C++",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    zip_safe=False,
    test_suite="pytest",
    tests_require=["pytest"],
)
