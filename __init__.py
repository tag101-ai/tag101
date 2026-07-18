"""Tag101 lightweight Bittensor subnet runtime."""

__all__ = ["SCOREBOARD_VERSION", "__version__", "semver_to_int_version"]

__version__ = "0.3.0"


def semver_to_int_version(version: str) -> int:
    """Encode ``major.minor.patch`` as an integer understood by scoreboards."""
    try:
        major, minor, patch = (int(part) for part in version.split("."))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid semantic version: {version!r}") from exc

    if major < 0 or not 0 <= minor <= 999 or not 0 <= patch <= 99:
        raise ValueError(f"semantic version is outside scoreboard limits: {version!r}")
    return major * 100_000 + minor * 100 + patch


SCOREBOARD_VERSION = semver_to_int_version(__version__)
