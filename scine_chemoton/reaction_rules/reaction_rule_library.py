#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Any, List, Optional

from scine_chemoton.reaction_rules.distance_rules import (
    DistanceRuleSet,
    DistanceRuleOrArray,
    FunctionalGroupRule,
    DistanceRuleAndArray,
    SimpleDistanceRule,
    DistanceBaseRule
)


def _default_init_repr(cls_inst: Any) -> str:
    return f"{cls_inst.__class__.__name__}()"


class DefaultOrganicChemistry(DistanceRuleSet):
    """
    This is a collection of reaction rules that should cover most of organic chemistry.
    """

    def __init__(self):
        rules = {
            'O': True,
            'N': True,
            'C': DefaultOrganicChemistry.C(),
            'H': DefaultOrganicChemistry.H()
        }
        super().__init__(rules)

    def __repr__(self) -> str:
        return _default_init_repr(self)

    class C(DistanceRuleOrArray):
        """
        Collection of organic chemistry reaction rules for carbon atoms.
        """

        def __init__(self):
            rules = [
                SpNCX(0, 2), SpNCX(0, 1),  # sp1 sp2
                AllylicSp3X(0),  # X=C-CR3
                AcetalX(0),  # at least two N/O
                CarbonylX(1),  # alpha position to carbonyl/imine group
            ]
            super().__init__(rules)

        def __repr__(self) -> str:
            return _default_init_repr(self)

    class H(DistanceRuleOrArray):
        """
        Collection of organic chemistry reaction rules for hydrogen atoms.
        """

        def __init__(self):
            rules = [
                AmmoniumX(1),  # R3N-H
                SimpleDistanceRule('O', 1),  # R-OH
                AcetalX(2),  # 2 position to an acetal
                AllylicSp3X(1),  # H on an sp3 C next to an sp2 C -> H-shift, Acidic through carbonyl group etc.
            ]
            super().__init__(rules)

        def __repr__(self) -> str:
            return _default_init_repr(self)


class CarbonylX(DistanceRuleOrArray):
    """
    Carbonyl/imine group at a distance of x bonds.

    Parameters
    ----------
        x : int
            Distance in bonds.
        hetero_atoms : List[str]
            A list of hetero atoms that define the carbonyl/imine carbon. Default ['O', 'N']
    """

    def __init__(self, x: int, hetero_atoms: Optional[List[str]] = None):
        if hetero_atoms is None:
            self._hetero_atoms = ['O', 'N']
        self._x = x
        rules: List[DistanceBaseRule] = [FunctionalGroupRule(x, 'C', (3, 3), {element: 1}, False)
                                         for element in self._hetero_atoms]
        super().__init__(rules)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._x}, {self._hetero_atoms})"


class CHOlefinC(DistanceRuleOrArray):
    """
    Olefinic carbon atoms with only H and C substituents.
    """

    def __init__(self):
        rules = [FunctionalGroupRule(0, 'C', (3, 3), {'C': 2, 'H': 1}, True),  # C=CH-C
                 FunctionalGroupRule(0, 'C', (3, 3), {'C': 1, 'H': 2}, True)  # C=CH2
                 ]
        super().__init__(rules)

    def __repr__(self) -> str:
        return _default_init_repr(self)


class SpNCX(FunctionalGroupRule):
    """
    Sp^n hybridization of a carbon atom at a distance of x bonds.

    Parameters
    ----------
        x : int
            Distance in bonds.
        n : int
            Carbon hybridization.
    """

    def __init__(self, x: int, n: int):
        if not (0 < n < 4):
            raise RuntimeError("The carbon hybridization may only have a maximum of 3 and a minimum of 1")
        n_bonded = n + 1
        self._x = x
        self._n = n
        super().__init__(x, 'C', (n_bonded, n_bonded))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._x}, {self._n})"


class CarboxylX(FunctionalGroupRule):
    """
    Carboxyl group at a distance of x bonds.

    Parameters
    ----------
        x : int
            Distance in bonds.
    """

    def __init__(self, x: int):
        self._x = x
        super().__init__(x, 'C', (3, 3), {'O': 2}, False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._x})"


class AmidX(FunctionalGroupRule):
    """
    Amid group at a distance of x bonds.
    """

    def __init__(self, x: int):
        """
        Construct with the given distance

        Parameters
        ----------
            x : int
                Distance in bonds.
        """
        self._x = x
        super().__init__(x, 'C', (3, 3), {'N': 1, 'O': 1}, False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._x})"


class AcetalX(DistanceRuleOrArray):
    """
    Acetal at a distance of x bonds. The acetal may have N substituents instead of only O.
    """

    def __init__(self, x: int, hetero_atoms: Optional[List[str]] = None):
        """
        Construct with a given distance and optional hetero atoms

        Parameters
        ----------
            x : int
                Distance in bonds.
            hetero_atoms : Optional[List[str]]
                A list of hetero atoms that define the acetal carbon. Default ['O', 'N']
        """
        self._x = x
        if hetero_atoms is None:
            self._hetero_atoms = ['O', 'N']
        rules: List[DistanceBaseRule] = [FunctionalGroupRule(x, 'C', (4, 4), {element: 2}, False)
                                         for element in self._hetero_atoms]
        n_hetero_atoms = len(self._hetero_atoms)
        for i in range(n_hetero_atoms):
            e1 = self._hetero_atoms[i]
            for j in range(i):
                e2 = self._hetero_atoms[j]
                rules.append(FunctionalGroupRule(x, 'C', (4, 4), {e1: 1, e2: 1}, False))
        super().__init__(rules)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._x}, {self._hetero_atoms})"


class CarboxylH(DistanceRuleAndArray):
    """
    Carboxyl H atom.
    """

    def __init__(self):
        super().__init__([SimpleDistanceRule('O', 1), CarboxylX(2)])

    def __repr__(self) -> str:
        return _default_init_repr(self)


class AmidH(DistanceRuleAndArray):
    """
    Amid H atom.
    """

    def __init__(self):
        super().__init__([SimpleDistanceRule('O', 1), AmidX(2)])

    def __repr__(self) -> str:
        return _default_init_repr(self)


class AllylicSp3X(DistanceRuleAndArray):
    """
    Allylic Sp^3 carbon at a distance of x bonds.
    Note that this rule should only be used with x = 1 for H or x = 0 for C, because otherwise it may lead
    to unintended results since the rule does not ensure that the sp^2 and sp^3 carbon atoms are neighbours.
    """

    def __init__(self, x: int):
        """
        Construct with the given distance

        Parameters
        ----------
            x : int
                Distance in bonds.
        """
        self._x = x
        rules: List[DistanceBaseRule] = [SpNCX(x, 3), SpNCX(x + 1, 2)]
        super().__init__(rules)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._x})"


class AmmoniumX(FunctionalGroupRule):
    """
    Ammonium N at a distance of x bonds.
    """

    def __init__(self, x: int):
        """
        Construct with the given distance

        Parameters
        ----------
            x : int
                Distance in bonds.
        """
        self._x = x
        super().__init__(x, 'N', (4, 4), {'N': 0, 'O': 0}, True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._x})"


class AminX(FunctionalGroupRule):
    """
    Amin group at a distance of x bonds.
    """

    def __init__(self, x: int):
        """
        Construct with the given distance

        Parameters
        ----------
            x : int
                Distance in bonds.
        """
        self._x = x
        super().__init__(x, 'N', (3, 3), {'N': 0, 'O': 0}, True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._x})"
