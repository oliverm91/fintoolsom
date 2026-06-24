# Plan de orden del proyecto fintoolsom

Estado actual relevante:
- Empaquetado: `setup.py` + `setup.cfg` + `requirements.txt` (este último solo apunta al propio repo vía git).
- Tests viven dentro de `src/tests/` (se empaquetarían si no fuera por `exclude = tests` en `setup.cfg`), usan `assert` + `print` a mano, no pytest, y se ejecutan a través de `run_tests.py` / `run_all_tests.py` / `setup_tests.py`.
- Dependencias externas problemáticas en tests:
  - `src/tests/create_bonds.py` lee un bono real desde `src/tests/fixed_income.db` (sqlite) vía `sqlite3`.
  - `src/tests/option_tests.py` carga `mkt_data.pkl` (superficie de vol, spot, curvas) con `pickle`.
- No existe carpeta `tests/` en la raíz; todo está bajo `src/tests/`.

## 1. Migrar empaquetado a `uv` + `pyproject.toml`
- Crear `pyproject.toml` (build-backend `hatchling` + `hatch-vcs`, igual a lo ya usado antes en el historial del repo) con:
  - `[project]`: nombre, autor, descripción, license, classifiers, `requires-python = ">=3.11"`, dependencias (`numpy>=2.0,<3.0`, `pandas>=2.0,<3.0`, `scipy==1.14`, `holidays>=0.58`, `numba`).
  - `[project.optional-dependencies]` o grupo `dev` con `pytest`.
  - `[tool.hatch.build.targets.wheel] packages = ["src/fintoolsom"]`.
  - `version` resuelto desde `src/fintoolsom/__version__.py` (mantener el archivo) o vía `hatch-vcs` a partir de tags git (decidir cuál; recomendado: mantener `__version__.py` para no depender de tags).
- Eliminar `setup.py`, `setup.cfg`, `requirements.txt`.
- Generar `uv.lock` con `uv lock` y dejar `uv.sync` como flujo de instalación documentado en `README.md`.
- Actualizar `README.md` con instrucciones de instalación/desarrollo usando `uv` (`uv sync`, `uv run pytest`).

## 2. Mover tests fuera de `src/`
- Crear `tests/` en la raíz del proyecto (hermano de `src/`).
- Mover y renombrar archivos de `src/tests/` a `tests/`:
  - `date_tests.py` → `tests/test_dates.py`
  - `rate_tests.py` → `tests/test_rates.py`
  - `option_tests.py` → `tests/test_options.py`
  - `create_bonds.py` → contenido reubicado como fixture/helper dentro de `tests/test_bonds.py` (ver punto 3), no como módulo aparte salvo que se prefiera `tests/conftest.py`.
- Eliminar `src/tests/run_tests.py`, `run_all_tests.py`, `setup_tests.py` (ver punto 4).
- Añadir `tests/__init__.py` solo si se requiere para imports relativos; si se usa pytest puro, no es necesario.
- Confirmar en `pyproject.toml` (`[tool.pytest.ini_options]`) `testpaths = ["tests"]` y que `tests/` quede fuera del paquete distribuido (ya no hace falta `exclude` porque vive fuera de `src/`).

## 3. Quitar dependencia de `fixed_income.db` y `mkt_data.pkl`
- **Bonos (`create_bonds.py` / `fixed_income.db`):**
  - Construir a mano, dentro de `tests/test_bonds.py`, un `CLBond` de ejemplo (2-3 cupones fijos, fechas y montos literales) que reemplace la consulta SQL contra `BTP0470930`.
  - Eliminar `fixed_income.db` del repo.
- **Opciones (`mkt_data.pkl`):**
  - Reconstruir a mano en `tests/test_options.py` el `DataFrame` de superficie de vol (`vol_surf_df`), `spot`, `local_zcc`, `foreign_zcc` usando los mismos valores que ya aparecen literalmente en `run_tests.py` (curva de dfs) y valores de vol fijos razonables (ej. todo en 0.15 como en `run_tests.py`), o una función auxiliar `build_sample_vol_surface()` en `tests/conftest.py`.
  - Eliminar `mkt_data.pkl` del repo.
- Ejecutar ambos tests tras la reconstrucción para confirmar que pasan sin los archivos binarios.

## 4. Eliminar runners manuales, migrar a pytest
- Convertir cada `run_tests()` en funciones `test_*` (una o varias por archivo) que usen `assert` (pytest los recolecta igual) en vez de `print('... Ok')`.
- Borrar `run_tests.py`, `run_all_tests.py`, `setup_tests.py` (innecesarios con pytest + `uv run pytest`).
- Verificar que `tests/test_options.py`, `tests/test_dates.py`, `tests/test_rates.py`, `tests/test_bonds.py` corren con `uv run pytest -q` sin imports rotos (ojo: `rate_tests.py` usa `RateConvention(InterestConvention.Linear, ...)` posicional con `InterestConvention.Linear`, que ya no existe en `Rates.py` actual — actualizar a `RateConvention(interest_convention=LinearInterestConvention, day_count_convention=ActualDayCountConvention, time_fraction_base=...)`).

## 5. Expandir cobertura de tests (máx. 10 métodos nuevos, solo los más críticos)
Priorizar métodos núcleo sin cobertura actual, evitando sobre-alcance:
1. `Bond.get_irr_from_amount` (o equivalente en `Bonds.py`) — ya se usa en `run_tests.py`, falta como test real.
2. `Bond.get_present_value` con `Rate` (IRR).
3. `Bond.get_present_value_zc` con `ZeroCouponCurve`.
4. `Bond.get_dv01`.
5. `Bond.get_z_spread`.
6. `ZeroCouponCurve` construcción desde `date_dfs` y consulta de un punto interpolado.
7. `Rate.convert_rate_conventions` (ida y vuelta entre dos convenciones).
8. `Call.get_mtm` / `Put.get_mtm` (uno de los dos, no ambos, para no duplicar `option_tests.py`).
9. `Tenor` / `get_day_count` con convención distinta a `Actual` (ej. `Days30E`), no cubierta hoy.
10. `NelsonSiegelSvensson` calibración básica (dado que es el módulo más reciente y sin tests) — *si el módulo aún no está mergeado a `main`, omitir este punto hasta que exista en la rama*.

No agregar más de estos 10; dejar explícitamente fuera: forwards, swaps, market/currencies, CLBonds variantes adicionales, eSSVI (ya cubierto parcialmente por `option_tests.py` existente).

## Orden de ejecución sugerido
1. Mover tests a `tests/` y limpiar runners (puntos 2 y 4) — esto no requiere tocar empaquetado todavía.
2. Reconstruir fixtures sin `.db`/`.pkl` (punto 3) y confirmar que todo corre con `python -m pytest` (entorno actual) antes de tocar el empaquetado.
3. Migrar a `pyproject.toml` + `uv` (punto 1), validar `uv run pytest`.
4. Añadir los tests nuevos priorizados (punto 5).
5. Actualizar `README.md` y borrar archivos obsoletos (`setup.py`, `setup.cfg`, `requirements.txt`, `fixed_income.db`, `mkt_data.pkl`).
