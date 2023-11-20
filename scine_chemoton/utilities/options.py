#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from scine_chemoton.utilities.comparisons import attribute_comparison


class BaseOptions:

    __slots__ = ()

    def __eq__(self, other) -> bool:
        return attribute_comparison(self, other)

    def unset_collections(self) -> None:
        """
        Duplicate name to HoldCollections method to be triggered in pickling process, so infinite _parent loops
        are avoided.
        """
        if hasattr(self, "_parent"):
            setattr(self, "_parent", None)
