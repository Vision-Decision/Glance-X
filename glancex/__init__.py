"""Glance-X: one model, three ways to look. Find objects from a text request: image-first, text-first or joint."""
__version__ = "0.2.0"


def __getattr__(name):  # keep `import glancex` light: torch / transformers load on first use
    if name in ("GlanceX", "Result", "ImageBank"):
        from . import inference
        return getattr(inference, name)
    raise AttributeError(name)


__all__ = ["GlanceX", "ImageBank", "Result", "__version__"]
