# Python 3.11 / 3.12 dependency-resolution failures (test_matrix.bat)

These combinations failed at `uv pip install` (dependency resolution), not at build or
test time — i.e. the pinned library versions are mutually incompatible, independent of
fintoolsom's code. Grouped by root cause (from `uv`'s resolver error messages).

## 1. `scipy==1.18.0` requires `numpy>=2.0.0` — conflicts with `numpy==1.26.4`
- python=3.11 numpy==1.26.4 pandas==1.5.3 scipy==1.18.0 (numba 0.59.1/0.60.0/0.65.1)
- python=3.11 numpy==1.26.4 pandas==2.2.3 scipy==1.18.0 (numba 0.59.1/0.60.0/0.65.1)
- python=3.11 numpy==1.26.4 pandas==3.0.3 scipy==1.18.0 (numba 0.59.1/0.60.0/0.65.1)
- python=3.12 numpy==1.26.4 pandas==1.5.3 scipy==1.18.0 (numba 0.59.1/0.60.0)
- python=3.12 numpy==1.26.4 pandas==2.2.3 scipy==1.18.0 (numba 0.59.1/0.60.0/0.65.1)
- python=3.12 numpy==1.26.4 pandas==3.0.3 scipy==1.18.0 (numba 0.59.1/0.60.0/0.65.1)

## 2. `numpy==2.5.0` requires `Python>=3.12` — conflicts with the python=3.11 venv
All `numpy==2.5.0` combos on python=3.11 (any pandas/scipy/numba): 27 combos total
(pandas 1.5.3/2.2.3/3.0.3 × scipy 1.11.4/1.14.1/1.18.0 × numba 0.59.1/0.60.0/0.65.1).

## 3. `numpy==2.5.0` vs. pinned `scipy`/`numba` upper bounds (on python=3.12, where the
   Python-version constraint from #2 no longer applies)
- `scipy==1.11.4` depends on `numpy>=1.21.6,<1.28.0` → conflicts with `numpy==2.5.0`
- `scipy==1.14.1` has a similar `numpy<2.x`-adjacent cap → conflicts with `numpy==2.5.0`
- `numba==0.59.1` depends on `numpy>=1.22,<1.27` → conflicts with `numpy==2.5.0`
- `numba==0.60.0` has a similar `numpy<2.x` cap → conflicts with `numpy==2.5.0`
- Net effect: every `numpy==2.5.0` combo on python=3.12 fails too (27 combos), because
  at least one of scipy/numba always pins numpy below 2.5.

**Takeaway:** `numpy==2.5.0` was too aggressive a pin for the "numpy 2.x" matrix slot —
none of the chosen scipy (1.11.4/1.14.1) or numba (0.59.1/0.60.0) pins support it, and
even `scipy==1.18.0` requires numpy 2.x but isn't itself installable on python<3.13 in
most of these combos. A more compatible "numpy 2.x" pin (e.g. `numpy==2.2.6`) would
resolve against the existing scipy/numba pins and remove most of this category.
