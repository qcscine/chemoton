# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

"""
This script creates a Pathfinder graph from an RMS input file.
Requires scine_chemoton.
"""


import scine_database as db
from scine_chemoton.gears.pathfinder import Pathfinder

manager = db.Manager()
db_name = "default"
credentials = db.Credentials("localhost", 27017, db_name)
manager.set_credentials(credentials)
manager.connect()

pathfinder = Pathfinder(manager)
pathfinder.options.graph_handler = "from-rms-input"
pathfinder.options.rms_file_name = "chem.rms"
pathfinder.build_graph()
pathfinder.export_graph("pathfinder.graph.json")
