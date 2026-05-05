"""Single-source package metadata helpers for SunXCorr.

The project metadata lives in ``pyproject.toml``. These helpers read the
installed distribution metadata so runtime code does not need hardcoded
version or author strings.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, metadata as distribution_metadata
from importlib.metadata import version as distribution_version

DIST_NAME = "sunxcorr"
FALLBACK_VERSION = "0.0.0"
FALLBACK_AUTHOR = "Slimane Mzerguat"
FALLBACK_DESCRIPTION = "Solar image coalignment utilities and correlation tools"


def _get_distribution_metadata() -> tuple[object | None, str]:
    try:
        return distribution_metadata(DIST_NAME), distribution_version(DIST_NAME)
    except PackageNotFoundError:
        return None, FALLBACK_VERSION


def get_version() -> str:
    """Return the installed package version."""

    _, version = _get_distribution_metadata()
    return version


def get_author() -> str:
    """Return the author metadata as a display string."""

    metadata, _ = _get_distribution_metadata()
    if metadata is None:
        return FALLBACK_AUTHOR

    authors = metadata.get_all("Author-email") or []
    authors += metadata.get_all("Maintainer-email") or []
    if not authors:
        authors = metadata.get_all("Author") or []
        authors += metadata.get_all("Maintainer") or []
    if authors:
        unique_authors = list(dict.fromkeys(authors))
        return ", ".join(unique_authors)
    return FALLBACK_AUTHOR


def get_description() -> str:
    """Return the short package summary."""

    metadata, _ = _get_distribution_metadata()
    if metadata is None:
        return FALLBACK_DESCRIPTION

    summary = metadata.get("Summary")
    return summary or FALLBACK_DESCRIPTION
