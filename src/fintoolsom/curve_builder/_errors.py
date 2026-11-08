"""Curve-builder errors and warnings."""


class InsufficientQuotesError(Exception):
    """A curve group is under-determined: more free pillars than instruments."""


class RepricingError(Exception):
    """A quote that pins a pillar no longer reprices once the whole build is done."""


class RedundantQuoteWarning(UserWarning):
    """Quotes ditched by the build: they own no pillar, so the curves ignore them.

    A quote is redundant when every curve it could pin is already built past its
    maturity by quotes solved earlier (e.g. a 1Y XCCY inside the FX forwards' range
    of the USD funding curve). It is valued on the final curves but not fitted.

    Attributes
    ----------
    ditched : list of tuple
        ``(quote, mtm, mtm_pct, mtm_bp)`` per ditched quote, in the report currency:

        * ``mtm``: its MTM on the final curves;
        * ``mtm_pct``: FX forwards only, MTM over the base-currency notional (None for
          a swap);
        * ``mtm_bp``: swaps only, MTM over its DV01, the MTM change for +1bp on the
          quoted rate (fixed rate, or the spread of a float-float swap): how many bp
          the quote is off par (None for an FX forward).
    """

    def __init__(self, message: str, ditched: list):
        super().__init__(message)
        self.ditched = ditched
