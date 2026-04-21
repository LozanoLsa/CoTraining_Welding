# 14 — Laser Weld Defect Detection · Co-Training (Semi-Supervised)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/CoTraining_Welding/blob/main/14_CoTraining_Welding.ipynb)

> *"Every weld tells two stories simultaneously — what the machine commanded, and what the metal answered back. Co-Training listens to both at once."*

---

## 🎯 Business Problem

On a laser welding line for automotive body panels, two independent data streams run simultaneously — every weld, every shift, without exception.

**View A** captures the *process conditions* set by the machine controller: laser pulse energy, travel speed, pulse frequency, spot diameter, shielding gas flow. These are the inputs — the parameters the operator can dial.

**View B** captures the *material response* measured by in-process sensors: bead height from laser profilometry, thermal variance from the pyrometer, acoustic emission RMS, penetration depth from ultrasound, vibration signature during deposition. These are the consequences — what the weld actually became.

Both views describe the same physical event. But they describe it from entirely different physical measurement principles, collected by independent instrument chains, without sharing a single sensor. That structural independence is not just a dataset curiosity — it is the exact condition that makes Co-Training possible.

The problem: of 1,500 welds in the dataset, only **150 have been manually inspected and labeled**. Full destructive testing or CT inspection on every weld is prohibitively slow and expensive in production. The remaining 1,350 welds went through the line with both sensor streams recording, but no human ever stamped them conforming or defective.

Co-Training exploits this asymmetry. Rather than discarding the 1,350 unlabeled welds, it teaches the two views to mentor each other: when View A is confident about a weld's quality, it donates that pseudo-label to expand View B's training set — and vice versa. Neither model teaches itself. The cross-view constraint is what separates Co-Training from Self-Training and what keeps the error from compounding inside a single model's blind spots.

---

## 📊 Dataset

**1,500 laser weld records · 10 features · 2 views · Target: `defect_label` (0 / 1 / NaN)**

Of the 150 labeled welds: 105 conforming (70%), 45 defective (30%). The 1,350 unlabeled welds carry full sensor readings but no quality stamp.

**View A — Process Parameters (machine controller output):**

| Column | Units | Description |
|--------|-------|-------------|
| `energy_j` | J | Laser pulse energy |
| `speed_mm_s` | mm/s | Welding travel speed |
| `frequency_khz` | kHz | Pulse repetition frequency |
| `spot_diameter_mm` | mm | Focused beam spot diameter |
| `gas_flow_l_min` | L/min | Shielding gas flow rate |

**View B — In-Process Sensor Monitoring (material response):**

| Column | Units | Description |
|--------|-------|-------------|
| `bead_height_mm` | mm | Weld bead height — laser profilometry |
| `thermal_var_c` | °C | Pyrometer thermal variance |
| `acoustic_rms` | — | Acoustic emission RMS signal |
| `penetration_mm` | mm | Weld penetration depth — ultrasound |
| `vibration_rms` | — | Vibration RMS during deposition |

**Data Origin — where these streams live in a real welding cell:**

| Feature | Source System | Instrument / Interface |
|---------|--------------|----------------------|
| `energy_j`, `speed_mm_s`, `frequency_khz`, `spot_diameter_mm` | Laser Controller (IPG / TRUMPF) | CAN bus → MES |
| `gas_flow_l_min` | Shielding Gas MFC | Flow controller → SCADA |
| `bead_height_mm` | Laser Profilometer (Keyence / Cognex) | Real-time metrology stream |
| `thermal_var_c` | Pyrometer (Raytek / LumaSense) | Process monitoring OPC-UA |
| `acoustic_rms`, `vibration_rms` | Piezo sensors on fixture | DAQ → historian |
| `penetration_mm` | Inline ultrasound (Olympus) | NDT stream → quality MES |

**Three EDA findings that shaped the model:**

**1. High travel speed is the dominant defect driver in View A.** Among labeled welds, faster-than-nominal speed consistently correlates with insufficient energy density per unit length — the thermal budget drops below what the joint geometry requires. The model's standardized coefficient for `speed_mm_s` (1.07) is the highest across both views.

**2. Penetration depth and bead height are the most protective sensor signals.** Both carry large negative coefficients in View B (-1.47 and -1.05 respectively). A weld that achieved full penetration and proper bead geometry is unlikely to be defective — the physics encoded themselves into the sensor record.

**3. Thermal variance separates quality classes more cleanly than any other View B feature.** Defective welds show higher pyrometer variance, consistent with unstable melt pool dynamics during deposition. This is the sensor the process engineer would have pointed to first.

---

## 🤖 Model — Co-Training with Logistic Regression

Co-Training is not a learning algorithm — it is a training protocol. It requires two learners that observe the same instances through genuinely independent feature sets and can produce calibrated probability estimates for their mutual teaching rounds.

Logistic Regression was chosen as the base learner for both views. It produces well-calibrated probabilities by design, which matters because the confidence threshold (0.65) is applied directly to its probability outputs. A poorly calibrated model would produce pseudo-labels at unpredictable quality levels regardless of the threshold chosen.

Each view has its own independent StandardScaler embedded in a Pipeline, preventing any information leakage between views during training or pseudo-labeling. `class_weight='balanced'` corrects for the 30% defect rate in the labeled set — without it, the models collapse to all-negative predictions as the Co-Training expansion dilutes the defect signal.

**Co-Training loop — what happened across 5 rounds:**

| Round | Pool A size | Pool B size | New to A | New to B |
|-------|------------|------------|---------|---------|
| Start | 120 | 120 | — | — |
| 1 | 897 | 237 | +777 | +117 |
| 2 | 1,167 | 430 | +270 | +193 |
| 3 | 1,249 | 465 | +82 | +35 |
| 4 | 1,257 | 467 | +8 | +2 |
| 5 | 1,262 | 467 | +5 | +0 |

Round 1 dominates: Model A (process view) teaches Model B 777 pseudo-labels in a single pass — its stronger initial signal on the cleaner process parameter space bootstraps the slower-learning sensor model rapidly. By Round 4 the loop converges; new teaching drops to single digits.

**A note on view independence:** The cross-view correlation analysis showed a maximum pairwise correlation of 0.846 between View A and View B features. This is higher than the ideal low-correlation scenario Co-Training textbooks describe. In practice, physical process–response coupling means some correlation is unavoidable on a real welding line. The models still benefited from mutual teaching because the *information content* of each view — what geometry in feature space each model carved — remained distinct even where individual features correlated.

---

## 📈 Key Results

Evaluated on 30 held-out real-labeled welds — test set was separated before any Co-Training round began. Pseudo-labels never entered evaluation.

| Model | Accuracy | AUC-ROC | F1 | Recall | Precision |
|-------|---------|---------|-----|--------|-----------|
| Model A — Process | 70.0% | 0.8095 | 0.609 | 77.8% | 50.0% |
| Model B — Sensors | 73.3% | 0.7725 | 0.636 | 77.8% | 53.8% |
| **Ensemble (avg)** | **73.3%** | **0.7884** | **0.636** | **77.8%** | **53.8%** |

**Confusion Matrix — Ensemble (n = 30 test records):**

| | Predicted: OK | Predicted: Defect |
|--|--|--|
| **Actual: OK** | 15 ✓ | 6 ✗ |
| **Actual: Defect** | 2 ✗ | 7 ✓ |

**Operational interpretation:** Recall of 77.8% means the ensemble caught 7 of 9 actual defective welds in the test set — the 2 misses (false negatives) represent welds that would reach downstream assembly undetected. On an automotive body panel line, false negatives carry direct cost in rework, warranty claims, and crash safety implications. The 6 false positives are over-triggers — welds flagged for inspection that were actually conforming. In a production setting, false positives cost inspection time; false negatives cost structural integrity.

The test set (30 records) is small enough to note: single-weld swings can move metrics meaningfully. AUC-ROC of 0.789 on a 10%-labeled problem is the more stable signal — it reflects the ensemble's ranking ability across the full probability range.

---

## 🔍 Top Drivers — Standardized Coefficients

**View A — Process Parameters (Model A):**

| Feature | Coefficient | Direction | Interpretation |
|---------|------------|-----------|----------------|
| `speed_mm_s` | +1.07 | ↑ defect risk | Faster travel → lower energy density → incomplete fusion |
| `energy_j` | −0.30 | ↓ defect risk | Higher pulse energy → more complete melt pool |
| `frequency_khz` | −0.26 | ↓ defect risk | Higher rep rate → more uniform heat distribution |
| `spot_diameter_mm` | −0.11 | ↓ defect risk | Wider spot → lower power density, more controlled |
| `gas_flow_l_min` | −0.03 | ↓ defect risk | Shielding gas contributes minimally in this range |

**View B — Sensor Response (Model B):**

| Feature | Coefficient | Direction | Interpretation |
|---------|------------|-----------|----------------|
| `penetration_mm` | −1.47 | ↓ defect risk | Full penetration = joint achieved — the strongest conformance signal |
| `bead_height_mm` | −1.05 | ↓ defect risk | Proper bead geometry indicates stable melt pool |
| `thermal_var_c` | +0.72 | ↑ defect risk | High pyrometer variance = unstable thermal front |
| `vibration_rms` | +0.66 | ↑ defect risk | Excessive vibration during deposition disrupts solidification |
| `acoustic_rms` | −0.35 | ↓ defect risk | Moderate acoustic signal associated with stable arc/plasma |

The two views tell a coherent physical story from opposite directions: defects happen when the process runs too fast with insufficient energy (View A) and manifests as poor penetration, low bead height, and high thermal instability (View B). Co-Training learned both sides of that story simultaneously from 150 labeled examples.

---

## 🗂️ Repository Structure

```
CoTraining_Welding/
│
├── 14_CoTraining_Welding.ipynb     # Notebook (no outputs)
├── welding_data.csv                 # 400-row sample (150 labeled + 250 unlabeled)
├── requirements.txt
└── README.md
```

> 📦 **Full Project Pack** — complete dataset (1,500 welds with all 1,350 unlabeled records),
> notebook with full outputs, presentation deck (PPTX + PDF), and `app.py` simulator
> available on [Gumroad](https://lozanolsa.gumroad.com).
>
> The GitHub CSV includes the full 150 labeled welds plus 250 unlabeled records — enough
> to run both views of the Co-Training pipeline and observe the pseudo-labeling mechanism.
> The full unlabeled pool (1,350 welds) showing all 5 convergence rounds is in the paid pack.

---

## 🚀 How to Run

**Google Colab (recommended):**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/CoTraining_Welding/blob/main/14_CoTraining_Welding.ipynb)

The notebook loads the CSV automatically from GitHub if not found locally.

**Local:**

```bash
git clone https://github.com/LozanoLsa/CoTraining_Welding.git
cd CoTraining_Welding
pip install -r requirements.txt
jupyter notebook 14_CoTraining_Welding.ipynb
```

**requirements.txt:**
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 💡 Key Learnings

**1. The value of Co-Training is structural, not statistical.**
It works because the two views are collected by independent instruments measuring different physical quantities. The algorithm doesn't create that independence — it just uses it. Before applying Co-Training, verify that your views genuinely come from different measurement principles. If they share the same sensor chain, the cross-teaching degrades to noise.

**2. Model A taught Model B far more than the reverse.**
In Round 1, the process view generated 777 pseudo-labels versus 117 from the sensor view. This asymmetry makes physical sense: process parameters are set by the controller and are therefore cleaner and more predictable than real-time sensor signals subject to fixture variation, acoustic reflections, and thermal drift. The sensor view needed the process view's early confidence to bootstrap itself.

**3. Recall is the metric that matters on a safety-critical weld line.**
A 70–73% accuracy number sounds modest. But catching 7 of 9 defective welds with only 120 labeled training examples — a Recall of 77.8% — is operationally meaningful. False negatives on structural automotive welds have a cost profile that false positives never can. Design the threshold accordingly.

**4. Pseudo-label growth is not uniform — and that is expected.**
Round 1 added 894 pseudo-labels combined; Round 5 added 5. Rapid early expansion followed by sharp convergence is the normal Co-Training behavior: the models exhaust the easy cases quickly. The long tail of uncertain welds (those that neither view could label confidently) is not a failure — it is the algorithm accurately identifying the hardest cases.

**5. The confidence threshold is a production parameter, not a model constant.**
The 0.65 threshold used here was chosen for calibrated Logistic Regression. On a real line, this value should be calibrated against confirmed inspection outcomes over the first production months. Set it too low and pseudo-label noise floods the training set. Set it too high and the expansion stalls. Empirical calibration, not theoretical defaults, is the right approach.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
