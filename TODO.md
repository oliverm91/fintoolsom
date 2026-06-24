# Fixed Income

Contexto:
- CLBonds vienen siempre definidos con flujos base 100 (suma amortizaciones es 100).
- Fórmula de valorización asume base 100:
  - Precio debe ser base 100
  - Valor Par es base 100
  - Monto a pagar se calcula con estos, por eso se divide en 10.000 y multiplica por notional
- Por eso `CLBond.coupons` tiene un adjust_to_notional.

- [x] Se eliminó la clase `Bond`, toda la lógica vive en `CLBond` (`Bonds.py` fusionado a `CLBonds.py`). Ver `.claude/fixed_income_plan.md`.
- [x] `CLBond.__init__` ahora tiene parámetros explícitos con validación de tipo/valor en vez de `**kwargs`.
- [x] **Bug corregido:** `get_irr_from_amount` calculaba `initial_guess` mezclando escalas — usaba `tera_value` en base 100 contra `dv01` y `amount` ya escalados por `notional`. Fix: `tera_value` ahora se escala por `notional/100` antes de calcular `initial_guess`. Test de regresión en `tests/test_clbonds.py::test_irr_from_amount_round_trips_for_real_notional`.
- Analizar qué métodos y código limpiar

# Derivatives
## Instrumentos
Deben estar definidos:
- Forwards: Normales (solo FX), NDF (solo FX) y sobre índice (para forwards de UF).
- Swaps: IRS, OIS, Basis, Cross Currency basis, Cross Currency fijo flotante.
- Opciones: Solo sobre FX.

Solo guardan datos, inmutables. No calculan.

## Market
- Debe tener la información para poder valorizar instrumentos definidos en sección anterior
- Definir objeto mercado que guarda:
  - Fecha
  - Historia:
    - Valores UF (pasado y futuro ya que la UF es conocida hasta el 9).
    - Spots: por si un NDF ya fijo hace n días y paga en m días.
    - Índices tipo TermIndex (estilo IBOR o Term SOFR): por si un swap fija cupón flotante al inicio del cupón y debe calcularse el flujo.
    - OvernightIndexes: Pueden ser tasa (EFFR en US) o un índice que devenga (ICP Chile)
  - Cotizaciones:
    - Spots (cruce entre 2 monedas. debe ser inteligente en que si tiene EURUSD y USDSCLP debe saber entregar EURCLP o incluso CLPEUR)
    - Puntos forward o Precios forward (UF no usa puntos forward son precios directamente)
    - Cotizaciones swaps tipo (estas deben decir claramente localidad de la cotización, calendarios aplicables, moneda de patas, índices flotantes por pata, madurez, frecuencia, unidad del valor (basis, porcentaje), moneda de colateral, índice de tasa de remuneración del colateral (quizás no es necesario moneda si los índices de tasa ya están atados a una moneda), es decir todo para construír el swap), tipo de ajuste para construcción de calendario (stub):
      - IRS
      - OIS
      - Basis
      - Cross Currency basis  
  - Curvas:
    - De descuento de monedas (depende de localidad y colateral (o solo localidad y ajusta por colateral con *FD_indiceCol/FD_monColLocal?))
    - De proyección de índices (curvas cero para Term SOFR 1M). Evaluar objeto ForwardRateCurve para este tipo de curvas
    - Pueden alimentarse a partir de:
      - Las cotizaciones
      - De forma manual

- Debe ser flexible para solicitar al menos fecha, pero puede estar vacío del resto.
- Evaluar agregar a Cotizaciones e Historia Rates para valorizar `CLBond`s.
- Evaluar agregar cruva de descuento de bonos de tesorería chileno (curva BTP, curva BTU). Pueden construirse a partir de las cotizaciones mediante `models.NelsonSiegelSvenson`.



## Calculadora
Instrumentos no deben tener get_mtm. Debe haber una calculadora que reciba un instrumento y el mercado y lo valoriza.