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

**1,912 laser weld records · 10 features · 2 views · Target: `defect_label` (0 / 1 / NaN)**

Of the 150 labeled welds: 105 conforming (70%), 45 defective (30%). The 1,762 unlabeled welds carry full sensor readings but no quality stamp.

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

**1. High travel speed is the dominant defect driver in View A.** Among labeled welds, faster-than-nominal speed consistently correlates with insufficient energy density per unit length — the thermal budget drops below what the joint geometry requires. The model's standardized coefficient for `speed_mm_s` (0.36) is the highest in View A.

**2. Vibration and thermal variance are the dominant defect signals in View B.** Excessive vibration during deposition and pyrometer variance both carry positive coefficients (+0.34 and +0.16 respectively), consistent with unstable melt pool dynamics during solidification. Bead height and penetration depth are protective — their negative coefficients reflect welds where the geometry turned out right.

**3. Thermal variance shows higher values in defective welds in the labeled distribution.** Defective welds show higher pyrometer variance, consistent with unstable melt pool dynamics during deposition. In the final trained model, vibration RMS carries the dominant signal in View B — the sensor the process engineer would reach for when fixture stability is suspect.

---

## 🤖 Model — Co-Training with Logistic Regression

Co-Training is not a learning algorithm — it is a training protocol. It requires two learners that observe the same instances through genuinely independent feature sets and can produce calibrated probability estimates for their mutual teaching rounds.

Logistic Regression was chosen as the base learner for both views. It produces well-calibrated probabilities by design, which matters because the confidence threshold (0.65) is applied directly to its probability outputs. A poorly calibrated model would produce pseudo-labels at unpredictable quality levels regardless of the threshold chosen.

Each view has its own independent StandardScaler embedded in a Pipeline, preventing any information leakage between views during training or pseudo-labeling. `class_weight='balanced'` corrects for the 30% defect rate in the labeled set — without it, the models collapse to all-negative predictions as the Co-Training expansion dilutes the defect signal.

**Co-Training loop — what happened across 5 rounds:**

| Round | Pool A size | Pool B size | New to A | New to B |
|-------|------------|------------|---------|---------|
| Start | 120 | 120 | — | — |
| 1 | 1,414 | 1,089 | +1,294 | +969 |
| 2 | 1,414 | 1,103 | 0 | +14 |
| 3 | 1,414 | 1,103 | 0 | 0 |
| 4 | 1,414 | 1,103 | 0 | 0 |
| 5 | 1,414 | 1,103 | 0 | 0 |

Round 1 dominates: Model B (sensor view) teaches Model A 1,294 pseudo-labels in a single pass — the sensor signals for vibration and penetration produce a high-confidence boundary on the unlabeled pool, bootstrapping the process model rapidly. Model A simultaneously teaches Model B 969 pseudo-labels from the process parameter space. By Round 2 the loop is essentially converged; only 14 additional assignments occur across the remaining four rounds.

**A note on view independence:** The cross-view correlation analysis showed a maximum pairwise correlation of 0.846 between View A and View B features. This is higher than the ideal low-correlation scenario Co-Training textbooks describe. In practice, physical process–response coupling means some correlation is unavoidable on a real welding line. The models still benefited from mutual teaching because the *information content* of each view — what geometry in feature space each model carved — remained distinct even where individual features correlated.

---

## 📈 Key Results

Evaluated on 30 held-out real-labeled welds — test set was separated before any Co-Training round began. Pseudo-labels never entered evaluation.

| Model | Accuracy | AUC-ROC | F1 | Recall | Precision |
|-------|---------|---------|-----|--------|-----------|
| Model A — Process | 73.3% | 0.8413 | 0.6667 | 88.9% | 53.3% |
| Model B — Sensors | 60.0% | 0.7196 | 0.4000 | 44.4% | 36.4% |
| **Ensemble (avg)** | **80.0%** | **0.8413** | **0.7000** | **77.8%** | **63.6%** |

**Confusion Matrix — Ensemble (n = 30 test records):**

| | Predicted: OK | Predicted: Defect |
|--|--|--|
| **Actual: OK** | 17 ✓ | 4 ✗ |
| **Actual: Defect** | 2 ✗ | 7 ✓ |

**Operational interpretation:** Recall of 77.8% means the ensemble caught 7 of 9 actual defective welds in the test set — the 2 misses (false negatives) represent welds that would reach downstream assembly undetected. On an automotive body panel line, false negatives carry direct cost in rework, warranty claims, and crash safety implications. The 4 false positives are over-triggers — welds flagged for inspection that were actually conforming. In a production setting, false positives cost inspection time; false negatives cost structural integrity.

The test set (30 records) is small enough to note: single-weld swings can move metrics meaningfully. AUC-ROC of 0.841 on an 8%-labeled problem is the more stable signal — it reflects the ensemble's ranking ability across the full probability range.

---

## 🔍 Top Drivers — Standardized Coefficients

**View A — Process Parameters (Model A):**

| Feature | Coefficient | Direction | Interpretation |
|---------|------------|-----------|----------------|
| `speed_mm_s` | +0.36 | ↑ defect risk | Faster travel → lower energy density → incomplete fusion |
| `gas_flow_l_min` | −0.17 | ↓ defect risk | Shielding gas provides a protective signal at this operating range |
| `frequency_khz` | −0.13 | ↓ defect risk | Higher frequency → slightly more stable energy delivery |
| `energy_j` | −0.09 | ↓ defect risk | Higher pulse energy → more complete melt pool |
| `spot_diameter_mm` | −0.01 | ↓ defect risk | Near-zero — spot geometry absorbed by speed and energy terms |

**View B — Sensor Response (Model B):**

| Feature | Coefficient | Direction | Interpretation |
|---------|------------|-----------|----------------|
| `vibration_rms` | +0.34 | ↑ defect risk | Excessive vibration during deposition — the dominant instability signal |
| `thermal_var_c` | +0.16 | ↑ defect risk | Pyrometer variance consistent with unstable melt pool dynamics |
| `bead_height_mm` | −0.13 | ↓ defect risk | Proper bead geometry indicates a stable, complete melt pool |
| `penetration_mm` | −0.11 | ↓ defect risk | Full penetration = joint achieved — strong conformance indicator |
| `acoustic_rms` | −0.10 | ↓ defect risk | Acoustic signal aligns with penetration and bead quality |

The two views tell a coherent physical story from opposite directions: defects happen when the process runs too fast (View A) and manifest as excessive vibration and thermal instability, with suppressed bead height and penetration depth (View B). Co-Training learned both sides of that story simultaneously from 150 labeled examples.

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

> 📦 **Full Project Pack** — complete dataset (1,912 welds with all 1,762 unlabeled records),
> notebook with full outputs, presentation deck (PPTX + PDF), and `app.py` simulator
> available on [Gumroad](https://lozanolsa.gumroad.com).
>
> The GitHub CSV includes the full 150 labeled welds plus 250 unlabeled records — enough
> to run both views of the Co-Training pipeline and observe the pseudo-labeling mechanism.
> The full unlabeled pool (1,762 welds) showing all 5 convergence rounds is in the paid pack.

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

**2. Model B taught Model A more than the reverse — especially in Round 1.**
In Round 1, the sensor view generated 1,294 confident pseudo-labels versus 969 from the process view. The vibration and penetration signals in View B drew a high-confidence boundary on the unlabeled pool early, allowing the sensor model to bootstrap the process model rather than the other way around. Both views contributed substantially — the asymmetry reflects which physical signals were most separable in this dataset, not a structural weakness in either view.

**3. Recall is the metric that matters on a safety-critical weld line.**
A 73% accuracy number sounds modest. But catching 7 of 9 defective welds with only 120 labeled training examples — a Recall of 77.8% — is operationally meaningful. False negatives on structural automotive welds have a cost profile that false positives never can. Design the threshold accordingly.

**4. Pseudo-label growth is not uniform — and that is expected.**
Round 1 added 2,263 pseudo-label assignments combined; Rounds 2–5 added 14 total. Rapid early expansion followed by near-complete convergence is the normal Co-Training behavior: the models exhaust the easy cases in a single pass. The welds that neither view could label confidently after Round 1 are not a failure — they are the algorithm accurately identifying the hardest cases in the unlabeled pool.

**5. The confidence threshold is a production parameter, not a model constant.**
The 0.65 threshold used here was chosen for calibrated Logistic Regression. On a real line, this value should be calibrated against confirmed inspection outcomes over the first production months. Set it too low and pseudo-label noise floods the training set. Set it too high and the expansion stalls. Empirical calibration, not theoretical defaults, is the right approach.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
