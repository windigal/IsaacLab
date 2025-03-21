# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations."""

import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

ISAACLAB_ASSETS_METADATA = toml.load(os.path.join(ISAACLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_ASSETS_METADATA["package"]["version"]


##
# Configuration for different assets.
##

from .allegro import *
from .ant import *
from .anymal import *
from .cart_double_pendulum import *
from .cartpole import *
from .franka import *
from .humanoid import *
from .kinova import *
from .leju import *
from .leju_v1 import *
from .quadcopter import *
from .ridgeback_franka import *
from .sawyer import *
from .shadow_hand import *
from .unitree import *
from .universal_robots import *
