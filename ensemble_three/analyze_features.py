"""Lightweight shim: original implementation moved to
`ensemble_three.archived.analyze_features` to reduce top-level clutter.

Importing * from the archived module so existing imports continue to work:

    from ensemble_three.analyze_features import calculate_target_correlation

"""

from .archived.analyze_features import *

__all__ = [name for name in dir() if not name.startswith('_')]
