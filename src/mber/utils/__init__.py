__all__ = ["RegionSpec", "HotspotSpec", "parse_region"]


def __getattr__(name):
    if name in __all__:
        from .regions import HotspotSpec, RegionSpec, parse_region

        exports = {
            "RegionSpec": RegionSpec,
            "HotspotSpec": HotspotSpec,
            "parse_region": parse_region,
        }
        return exports[name]
    raise AttributeError(f"module 'mber.utils' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
