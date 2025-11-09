"""
Global / experimental configuration flags.
"""

from dataclasses import dataclass


@dataclass
class TBConfig:
    debug: bool = False


config = TBConfig()
