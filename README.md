You are a professional quantitative developer.

Build a crypto market making system in Python.

Structure:

* engine.py (Avellaneda-Stoikov model)
* alpha.py (alpha signal)
* features.py (feature engineering)
* backtest.py (backtesting)
* run.py (main)

Requirements:

1. Implement Avellaneda-Stoikov:

* reservation price
* spread
* inventory control

2. Alpha model:

* imbalance
* microprice
* spread

3. Backtest:

* simulate fills
* calculate pnl

Use only numpy and pandas.

Write clean and complete code.
## Step-by-Step Implementation Plan

### Step 1: Set Up Project Structure

- Create the following files:
  - `engine.py` — Implements the Avellaneda-Stoikov market making engine.
  - `alpha.py` — Implements alpha signal models (order book imbalance, microprice, etc.).
  - `features.py` — Feature engineering utilities (for input to alphas/engine).
  - `backtest.py` — Backtesting framework (simulate trades, fills, and calculate PnL).
  - `run.py` — Main script for running a simple experiment (loading data, running strategy).

### Step 2: Implement Avellaneda-Stoikov Market Making Engine (`engine.py`)

- Define a class to model:
  - Reservation price calculation.
  - Optimal bid/ask spread.
  - Inventory control mechanism.
- Use only `numpy` and `pandas` for calculations.
- Input: market signals, current inventory, risk aversion, etc.
- Output: recommended bid/ask prices.

### Step 3: Alpha Model (`alpha.py`)

- Implement functions for:
  - Order book imbalance signal.
  - Microprice calculation.
  - Spread features.
- All inputs/outputs are pandas Series or DataFrame.

### Step 4: Feature Engineering (`features.py`)

- Utilities for creating time-series features from raw market data:
  - Rolling statistics, moving averages, volatility, etc.
- Used by both alpha and engine modules.

### Step 5: Backtesting Framework (`backtest.py`)

- Simulate order submission, fills (with latency/slippage), PnL calculation.
- Inject synthetic or historical price/volume/order book data.
- Track inventory, quote changes, and trade history.

### Step 6: Main Script (`run.py`)

- Example pipeline: load historical data, generate features, signal generation, run engine/backtest, summarize results.

### Step 7: Test & Polish

- Write small test cases or scripts for each module.
- Carefully document code and functions for clarity.

---

Proceed to implement each step, ensuring each module is self-contained and importable. Start with data structures, then engine and alpha signals, followed by backtesting logic.

