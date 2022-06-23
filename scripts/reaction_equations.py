#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualizes all reactions in the database with molassembler graphs.
   Runs at least with the native IPython support of vscode"""

__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from cairosvg import svg2png
from io import BytesIO
from json import dumps
from graphviz import Source
from PIL import ImageTk, Image
from subprocess import call
from tqdm import tqdm
from typing import Union, Tuple

import os
import re
import sys
import numpy as np
import tkinter as tk

import scine_database as db
import scine_utilities as utils
import scine_molassembler as masm

"""
This script displays all reaction equations found by Chemoton in one given database in a GUI.
In the GUI the graphs can be clicked to open the structures with PyMOL.
This script is lazy in the sense that it is always executed when loaded and makes use of 'global' variables.
All of this could be resolved by proper oop changes and function handling.
"""

if os.environ.get("DISPLAY", "") == "":
    print("no display found. Using :0.0")
    os.environ.__setitem__("DISPLAY", ":0.0")

# Settings
db_name = "propylene-gas-pbe-d3-tzvp-first-shell-redo"
db_port = 27009
barrier_limit = 500
# model = db.Model("gfn2", "", "")
model = db.Model("dft", "pbe d3bj", "def2-TZVP")
model.spin_mode = "unrestricted"
show_ids = True
credentials = db.Credentials("127.0.0.1", db_port, db_name)

total_width = 1500
total_height = 1000  # irrelevant due to scrolling
arrow_size = 100

ARROW_STRING = """
<svg:svg
   xmlns:svg="http://www.w3.org/2000/svg"
   version="1.0"
   width="902.25049"
   height="364.71875"
   viewBox="0 0 902.25049 364.71875"
   id="svg2868"
   xml:space="preserve"><svg:defs
   id="defs2874" />
<namedview
   id="base"
   pagecolor="#ffffff"
   bordercolor="#666666"
   borderopacity="1.0"
   pageopacity="0.0"
   pageshadow="2"
   window-width="640"
   window-height="541"
   zoom="0.34493828"
   cx="372.04722"
   cy="256.66814"
   window-x="75"
   window-y="152"
   current-layer="svg2033">
</namedview>
<svg:g
   transform="matrix(-1,0,0,-1,902.25049,364.71875)"
   id="Ebene_1">
<svg:polygon
   points="902.25049,222.98633 233.17773,222.98633 233.17773,364.71875 0,182.35938 233.17773,0 233.17773,141.73242 902.25049,141.73242 902.25049,222.98633 "
   id="path2050" />
</svg:g>
</svg:svg>
"""
ARROW_PNG = svg2png(ARROW_STRING.encode("utf-8"), output_width=arrow_size, output_height=arrow_size)


def reaction_click(event, number):
    """
    Retrieve structures and open with pymol
    """
    idx = reaction_ids[number]
    print("Looking at reaction: " + str(idx))
    r = db.Reaction(idx)
    r.link(reactions)
    step = db.ElementaryStep(step_with_lowest_barrier(r))
    step.link(elementary_steps)
    ts = db.Structure(step.get_transition_state())
    ts.link(structures)
    structure_args = ["tmp_ts.xyz"]
    utils.io.write("tmp_ts.xyz", ts.get_atoms(), es_string(ts))
    for i, reactant in enumerate(step.get_reactants(db.Side.LHS)[0]):
        ed = db.Structure(reactant)
        ed.link(structures)
        name = "tmp_reactant_" + str(i) + ".xyz"
        structure_args.append(name)
        utils.io.write(name, ed.get_atoms(), es_string(ed))
    for i, reactant in enumerate(step.get_reactants(db.Side.RHS)[1]):
        ed = db.Structure(reactant)
        ed.link(structures)
        name = "tmp_product_" + str(i) + ".xyz"
        structure_args.append(name)
        utils.io.write(name, ed.get_atoms(), es_string(ed))
    call(["pymol"] + structure_args)
    call(["rm"] + structure_args)


def click(event, number):
    """
    Retrieve structure and open with pymol
    """
    idx = ids[number]
    print("Looking at compound: " + str(idx))
    c = db.Compound(idx)
    c.link(compounds)
    s = db.Structure(c.get_centroid())
    s.link(structures)
    utils.io.write("tmp.xyz", s.get_atoms(), es_string(s))
    call(["pymol", "tmp.xyz"])
    call(["rm", "tmp.xyz"])


def step_with_lowest_barrier(reaction: db.Reaction) -> db.ID:
    barriers = []
    for step_id in reaction.get_elementary_steps():
        step = db.ElementaryStep(step_id)
        step.link(elementary_steps)
        ts = db.Structure(step.get_transition_state())
        ts_energy = energy(ts, "gibbs_free_energy", model, structures, properties)
        if ts_energy is None:
            barriers.append(np.inf)
            continue
        try:
            reactant_energy = sum(
                energy(
                    db.Structure(reactant),
                    "gibbs_free_energy",
                    model,
                    structures,
                    properties,
                )
                for reactant in step.get_reactants(db.Side.LHS)[0]
            )
        except TypeError:
            barriers.append(np.inf)
            continue
        barriers.append(ts_energy - reactant_energy)
    return reaction.get_elementary_steps()[np.argmin(barriers)]


def es_string(structure: db.Structure) -> str:
    c = structure.get_charge()
    m = structure.get_multiplicity()
    return str(c) + " " + str(m)


def barrier_height(
    reaction: db.Reaction, elementary_steps, model, structures, properties
) -> Tuple[Union[float, None], Union[float, None]]:
    """
    Gives the lowest barrier height of the forward reaction (left to right) in kJ/mol out of all the elementary
    steps grouped into this reaction. Barrier height are given as Tuple with the first being gibbs free energy
    and second one the electronic energy. Returns None for not available energies

    Parameters
    ----------
    reaction : scine_database.Reaction (Scine::Database::Reaction)
        The reaction we want the barrier height from

    Returns
    -------
    Tuple[Union[float, None], Union[float, None]]
        barrier height in kJ/mol
    """
    barriers = {"gibbs_free_energy": [], "electronic_energy": []}
    for step_id in reaction.get_elementary_steps():
        step = db.ElementaryStep(step_id)
        step.link(elementary_steps)
        for energy_type, values in barriers.items():
            barrier = single_barrier(step, energy_type, model, structures, properties)
            if barrier is not None:
                values.append(barrier)
    gibbs = None if not barriers["gibbs_free_energy"] else min(barriers["gibbs_free_energy"])
    electronic = None if not barriers["electronic_energy"] else min(barriers["electronic_energy"])
    return gibbs, electronic


def single_barrier(step: db.ElementaryStep, energy_type: str, model, structures, properties) -> Union[float, None]:
    """
    Gives the barrier height of a single elementary step (left to right) in kJ/mol for the specified energy type.
    Returns None if the energy type is not available.

    Parameters
    ----------
    step : scine_database.ElementaryStep (Scine::Database::ElementaryStep)
        The elementary step we want the barrier height from
    energy_type : str
        The name of the energy property such as 'electronic_energy' or 'gibbs_free_energy'

    Returns
    -------
    Tuple[Union[float, None], Union[float, None]]
        barrier height in kJ/mol
    """
    ts = db.Structure(step.get_transition_state())
    ts_energy = energy(ts, energy_type, model, structures, properties)
    reactant_energy_generator = [
        energy(db.Structure(reactant), energy_type, model, structures, properties)
        for reactant in step.get_reactants(db.Side.LHS)[0]
    ]
    reactant_energy = None if None in reactant_energy_generator else sum(reactant_energy_generator)
    return None if None in [ts_energy, reactant_energy] else (ts_energy - reactant_energy) * utils.KJPERMOL_PER_HARTREE


def energy(
    structure: db.Structure,
    prop_name: str,
    model: db.Model,
    structures: db.Collection,
    properties_coll: db.Collection,
) -> Union[float, None]:
    """
    Gives energy value depending on demanded property. If the property does not exit, None is returned

    Parameters
    ----------
    structure : scine_database.Structure (Scine::Database::Structure)
        The structure we want the energy from
    prop_name : str
        The name of the energy property such as 'electronic_energy' or 'gibbs_free_energy'

    Returns
    -------
    Union[float, None]
        energy value in Hartree
    """
    structure.link(structures)
    properties = structure.query_properties(prop_name, model, properties_coll)
    if not properties:
        return None
    # pick last property if multiple
    prop = db.NumberProperty(properties[-1])
    prop.link(properties_coll)
    return prop.get_data()


def ratio(svg: str):
    """
    Get width and height from svg string generated by masm
    """
    ratio = re.findall("[0-9]+pt", svg)
    ratio = [int(s.replace("pt", "")) for s in ratio]
    return ratio


def extract_svg(structure: db.Structure):
    """
    Get the binary svg from a db structure
    """
    tmp_str = structure.get_graph("masm_cbor_graph")
    tmp_binary = masm.JsonSerialization.base_64_decode(tmp_str)
    masm_molecule = masm.JsonSerialization(tmp_binary, masm.JsonSerialization.BinaryFormat.CBOR).to_molecule()
    bin_svg = Source(masm_molecule.dump_graphviz()).pipe(format="svg")
    w, h = ratio(str(bin_svg))
    str_svg = re.sub(
        r"<text text\-anchor\=\"[a-z]+\" x\=\"-?\d*\.?\d*\" y\=\"-?\d*\.?\d*\" font\-family\=\"Arial\" font\-size\=\"\d*\.?\d*\" ?(?:fill\=\"[a-z]+\")?>\w*<\/text>",
        "",
        str(bin_svg),
    )
    str_svg = str_svg.replace(r"\n", "\n")
    str_svg = str_svg.replace(r"b'", "")
    str_svg = str_svg.replace(r"'", "")
    widths.append(w)
    heights.append(h)
    svgs.append(str_svg)


def add_reaction(reactants, heights, widths, svgs, height_offset, barriers):
    """
    Add reaction to GUI window
    """
    x_offset = 5

    # determine widths
    width_factor_left = 1.0 / len(reactants[0])
    width_factor_right = 1.0 / len(reactants[1])

    left_widths = [width_factor_left * (total_width - arrow_size - 2 * x_offset) / 2 - x_offset] * len(reactants[0])
    right_widths = [width_factor_right * (total_width - arrow_size - 2 * x_offset) / 2 - x_offset] * len(reactants[1])
    new_widths = left_widths + right_widths

    # determine heights
    new_heights = [nw * h / w for nw, h, w in zip(new_widths, heights, widths)]  # keep aspect ratio
    max_height = max(new_heights)  # used for centering everything on arrow

    # add arrow
    arrows.append(ImageTk.PhotoImage(Image.open(BytesIO(ARROW_PNG))))
    arrow_buttons.append(
        canvas.create_image(
            (total_width - arrow_size) / 2,
            0.5 * max_height - arrow_size / 2 + height_offset,
            anchor=tk.NW,
            image=arrows[-1],
        )
    )
    barrier_entry_0 = "?" if barriers[0] is None else int(round(barriers[0]))
    barrier_entry_1 = "?" if barriers[1] is None else int(round(barriers[1]))
    canvas.create_text(
        (total_width + arrow_size) / 2 - x_offset,
        0.5 * max_height - arrow_size / 2 + height_offset - 5,
        fill="darkblue",
        font="Times 10 bold",
        anchor=tk.NE,
        text="{} kJ/mol".format(barrier_entry_0),
    )
    canvas.create_text(
        (total_width + arrow_size) / 2 - x_offset,
        0.5 * max_height + arrow_size / 2 + height_offset - 5,
        fill="darkblue",
        font="Times 10 bold",
        anchor=tk.NE,
        text="{} kJ/mol".format(barrier_entry_1),
    )

    # add graphs
    x_pos = x_offset
    for count, (w, h, svg) in enumerate(zip(new_widths, new_heights, svgs)):
        arrow_offset = arrow_size + x_offset if count == len(reactants[0]) else 0
        png = svg2png(svg, output_width=w, output_height=h)
        images.append(ImageTk.PhotoImage(Image.open(BytesIO(png))))
        buttons.append(
            canvas.create_image(
                x_pos + arrow_offset,
                (max_height - h) / 2 + height_offset,
                anchor=tk.NW,
                image=images[-1],
            )
        )
        c_id = reactants[0][count] if count < len(reactants[0]) else reactants[1][count - len(reactants[0])]
        compound = db.Compound(c_id)
        compound.link(compounds)
        struc = db.Structure(compound.get_centroid())
        struc.link(structures)
        # add charge multiplicity
        canvas.create_text(
            x_pos + arrow_offset,
            (max_height - h) / 2 + height_offset,
            fill="darkblue",
            font="Times 10 bold",
            anchor=tk.NW,
            text="{}".format(es_string(struc)),
        )
        # add compound id
        if show_ids:
            canvas.create_text(
                x_pos + arrow_offset + w,
                (max_height - h) / 2 + height_offset,
                fill="darkblue",
                font="Times 10 bold",
                anchor=tk.NE,
                text=str(compound.id()),
            )
        x_pos += w + arrow_offset + x_offset
    return max_height + height_offset  # current highest y-value (lowest position)


# Database access
manager = db.Manager()
manager.set_credentials(credentials)
manager.connect()
calculations = manager.get_collection("calculations")
structures = manager.get_collection("structures")
properties = manager.get_collection("properties")
reactions = manager.get_collection("reactions")
elementary_steps = manager.get_collection("elementary_steps")
compounds = manager.get_collection("compounds")

# GUI definitions
window = tk.Tk()
window.geometry(str(total_width) + "x" + str(total_height))
window.title(db_name)
frame = tk.Frame(window, width=total_width, height=total_height)
frame.pack(expand=True, fill=tk.BOTH)
n_reaction = reactions.count(dumps({}))
canvas = tk.Canvas(
    window,
    width=total_width,
    height=total_height,
    scrollregion=(0, 0, n_reaction * 1000, n_reaction * 1000),
)
vbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
vbar.pack(side="right", fill="y")
canvas.config(width=total_width, height=total_height)
canvas.config(yscrollcommand=vbar.set)
canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
vbar.config(command=canvas.yview)

# global containers
images = []
buttons = []
arrow_buttons = []
arrows = []
ids = []
reaction_ids = []

lowest_point = 0
with tqdm(total=n_reaction, file=sys.stdout, desc="scanning reactions") as pbar:
    for reaction_count, reaction in tqdm(enumerate(reactions.iterate_all_reactions())):
        reaction.link(reactions)
        pbar.update(1)
        barriers = barrier_height(reaction, elementary_steps, model, structures, properties)
        if all(barrier > barrier_limit for barrier in barriers if barrier is not None):
            continue
        n_steps = len(reaction.get_elementary_steps())

        reactants = reaction.get_reactants(db.Side.BOTH)
        ids += reactants[0] + reactants[1]
        reaction_ids.append(reaction.id())

        # containers for extract_svg
        heights = []
        widths = []
        svgs = []
        for idx in reactants[0] + reactants[1]:
            reactant = db.Compound(idx)
            reactant.link(compounds)
            structure = db.Structure(reactant.get_centroid())
            structure.link(structures)
            extract_svg(structure)

        height_offset = lowest_point + 20
        lowest_point = add_reaction(reactants, heights, widths, svgs, height_offset, barriers)

# bind opening pymol action to images
for i in range(len(buttons)):
    canvas.tag_bind(buttons[i], "<Button-1>", lambda event, i=i: click(event, i))
for i in range(len(arrow_buttons)):
    canvas.tag_bind(arrow_buttons[i], "<Button-1>", lambda event, i=i: reaction_click(event, i))

# open window
window.mainloop()
