#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from json import dumps
from graphviz import Source
from tqdm import tqdm
from typing import List

import os
import re
import sys

import scine_database as db
import scine_utilities as utils
import scine_molassembler as masm

""" INPUT """
db_name = "puffin-test"
db_port = 27009
credentials = db.Credentials('127.0.0.1', db_port, db_name)
compound_sele = {}
#compound_sele = {'exploration_disabled': False}
svg_dir_name = "tmp-svgs"
html_name = "compounds.html"
""" INPUT END"""

def extract_svg(structure: db.Structure, svgs: List[str]):
    tmp_str = structure.get_graph('masm_cbor_graph')
    tmp_binary = masm.JsonSerialization.base_64_decode(tmp_str)
    masm_molecule = masm.JsonSerialization(tmp_binary, masm.JsonSerialization.BinaryFormat.CBOR).to_molecule()
    bin_svg = Source(masm_molecule.dump_graphviz()).pipe(format="svg")
    str_svg = re.sub(r"<text text\-anchor\=\"[a-z]+\" x\=\"-?\d*\.?\d*\" y\=\"-?\d*\.?\d*\" font\-family\=\"Arial\" font\-size\=\"\d*\.?\d*\" ?(?:fill\=\"[a-z]+\")?>\w*<\/text>", "", str(bin_svg))
    str_svg = str_svg.replace(r"\n", "\n")
    str_svg = str_svg.replace(r"b'", "")
    str_svg = str_svg.replace(r"'", "")
    svgs.append(str_svg.replace(r"<title>G", r"<title>" + str(structure.get_compound())))
    return svgs

# Database access
manager = db.Manager()
manager.set_credentials(credentials)
manager.connect()
structures = manager.get_collection('structures')
compounds = manager.get_collection('compounds')
n_compounds = compounds.count(dumps({}))

try:
	os.mkdir(svg_dir_name)
except:
	print("Already svgs present; rename or delete " + svg_dir_name + " directory first")
	sys.exit()
master_svgs = []
with tqdm(total=n_compounds, file=sys.stdout, desc="scanning compounds") as pbar:
	for compound in compounds.iterate_compounds(dumps(compound_sele)):
		compound.link(compounds)
		master_svgs = extract_svg(db.Structure(compound.get_centroid(), structures), master_svgs)
        	pbar.update(1)


for i, svg in enumerate(master_svgs):
	with open(os.path.join(svg_dir_name, str(i)+".svg"), "w") as f:
		f.write(svg)
with open(html_name, "w") as f:
	f.write("<html>\n<body>\n")
	for i, svg in enumerate(master_svgs):
		f.write("<embed src='"+str(os.path.join(svg_dir_name, str(i)+".svg")) + "' type='image/svg+xml' />\n")
	f.write("</body>\n</html>")
