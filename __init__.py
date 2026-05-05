"""Compatibility wrapper for the historical top-level SunXCorr module.

The packaged API lives in :mod:`sunxcorr`; this module keeps the repository
root importable without duplicating metadata or version information.
"""

from sunxcorr import *  # noqa: F401,F403
from sunxcorr._metadata import get_author, get_description, get_version

__version__ = get_version()
__author__ = get_author()
__description__ = get_description()
