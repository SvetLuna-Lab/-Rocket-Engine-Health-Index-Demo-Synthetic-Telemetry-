# Rocket Engine Health Index Demo (Synthetic Telemetry)

This repository contains a small, focused **AI-assisted health monitoring** demo  
for a **liquid-propellant rocket engine**.

The idea is simple:

- simulate realistic-looking engine telemetry (pressure, turbopump speed, fuel flow, vibrations, etc.),
- inject several **fault regimes** (pressure decay, turbopump overspeed, fuel leak, bearing overheat, …),
- train a classifier to distinguish **normal vs faulty** behavior,
- compute a **health index** over time:  
  > health_index ≈ P(engine is in a normal regime)

The goal is not to build a full PHM system, but to show a clean, readable workflow:
from synthetic telemetry → to features → to a model → to a health index time series
and plots that an engineer can interpret.

---

## Project structure

```text
rocket-engine-health-index-demo/
├─ data/
│  └─ .gitkeep                  # CSV files with synthetic telemetry & health_index will appear here
├─ figures/
│  └─ .gitkeep                  # PNG plots (time series, health index) will be saved here
├─ src/
│  ├─ __init__.py
│  ├─ generate_telemetry.py     # synthetic telemetry generator with multiple channels & fault modes
│  ├─ preprocessing.py          # rolling-window features for time series
│  ├─ train_health_index.py     # train classifier + compute health index = P(normal)
│  └─ visualization.py          # plotting utilities (time series, health index)
├─ README.md
├─ requirements.txt
└─ .gitignore



Telemetry channels and fault regimes

The synthetic telemetry is generated at 1 Hz over a typical engine run (e.g. 300 s).

Channels:

Pc – chamber pressure

N_pump – turbopump speed

T_in – injector inlet temperature

Vib – overall vibration level

fuel_flow – fuel mass flow rate

bearing_temp – bearing temperature

thrust – simplified thrust proxy (combination of pressure and flow)

Each run is labeled with a regime:

normal — nominal operation

pressure_decay — slow loss of chamber pressure

turbopump_overspeed — excessive turbopump speed in the second half of the run

temp_rise — strong injector temperature rise late in the run

vibration_increase — global vibration increase (e.g. imbalance / cavitation)

fuel_leak — fuel flow increases, while pressure and thrust degrade

bearing_overheat — bearing temperature rises together with vibrations

These patterns are not physically perfect, but they are qualitatively realistic:
they produce time series that “look and feel” like real telemetry and are useful
for demonstrating health monitoring logic.



Installation

Create and activate a virtual environment (optional but recommended),
then install the dependencies:

pip install -r requirements.txt


Usage
1. Generate synthetic telemetry

python -m src.generate_telemetry
# or
python src/generate_telemetry.py


This will create multiple CSV files in data/:

data/
  normal_run_0.csv
  normal_run_1.csv
  normal_run_2.csv
  pressure_decay_run_0.csv
  ...
  bearing_overheat_run_1.csv


Each file contains columns like:

time, Pc, N_pump, T_in, Vib, fuel_flow, bearing_temp, thrust, label


2. Train the classifier and compute the health index

python -m src.train_health_index
# or
python src/train_health_index.py


The script will:

Load all CSV files from data/.

Apply rolling-window feature engineering (mean/std over a small window)
for each telemetry channel.

Train a RandomForestClassifier to distinguish normal vs various fault labels.

Print a classification report on a validation split.

Compute the health index on the full feature set:

health_index(t) = P(class == "normal" | telemetry at time t)


Save the health index time series to:

data/health_index.csv


Plot the health index over time with a simple threshold line and save it to:

figures/health_index.png


Example console output (the exact numbers will vary):

Classification report (fault vs normal regimes):
              precision    recall  f1-score   support

      normal       0.97      0.96      0.96      2000
pressure_decay      ...
...

accuracy                           0.96      8000
macro avg                          ...
weighted avg                       ...

Saved health index time series to .../data/health_index.csv
Saved plot to figures/health_index.png


How the health index works

The classifier is trained as a multi-class model over regimes
(normal, pressure_decay, fuel_leak, etc.).

For each time step we take the model’s predicted probabilities and extract:

health_index(t) = P(class == "normal")


Intuitively:

health_index ≈ 1.0 — the engine looks very much like a normal run;

health_index drops towards 0 — the model “believes” that some fault regime is more likely.

On the plot you will see:

a curve of health_index over time,

a horizontal threshold (e.g. 0.7) indicating where the engine should be treated as “suspicious”.

This is intentionally simple, but it mirrors real PHM / condition monitoring logic:
convert raw telemetry into an interpretable health signal that engineers can track.


Possible extensions

If you want to evolve this demo further, natural next steps could be:

add more realistic dynamics to the synthetic telemetry (start/stop sequences, throttling, staging),

model sensor noise / dropouts / drifts,

add a remaining useful life (RUL) or time-to-failure prediction on top of the health index,

compare different models (Gradient Boosting, XGBoost, simple neural nets),

integrate with an experiment tracker (MLflow, Weights & Biases, etc.),

wrap the health index into a simple API or dashboard.

Even though the data here is synthetic, the core idea is close to real aerospace
health monitoring: turn raw engine telemetry into a stable, interpretable signal
that says how “healthy” the engine looks at any moment in time.
