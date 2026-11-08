"""Curve-builder errors."""


class InsufficientQuotesError(Exception):
    """A curve group is under-determined: more free pillars than instruments."""


class RepricingError(Exception):
    """A quote that pins a pillar no longer reprices once the whole build is done."""
