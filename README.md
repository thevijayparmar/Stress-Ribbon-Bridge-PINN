# 🌉 Explainable AI for Stress-Ribbon Bridges: PINN vs. Surrogate

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_Optimized-EE4C2C.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-FF4B4B.svg)
![Status](https://img.shields.io/badge/Status-Research_Ready-success.svg)

Welcome to the computational frontier of structural engineering! 🏗️

This repository contains the complete, end-to-end Machine Learning pipeline for designing and analyzing **Cable-Supported & Stress-Ribbon Bridges**. By replacing traditional, mathematically expensive iterative solvers with **Physics-Informed Neural Networks (PINNs)** and **Data-Driven Surrogate Models**, this toolset achieves up to a **33x computational speedup** with sub-0.55% Normalized Mean Absolute Error (NMAE).

Whether you are evaluating a pedestrian Short Span (≤ 300m) or a massive Long Span (≤ 2000m), this repository provides the mathematical generators, the AI training architectures, and the interactive UI to handle it.

---

## ✨ Key Features

- **Massive Data Generation:** Exact-math catenary solvers capable of generating over 17 million physically viable structural combinations.
- **Dual AI Architectures:** Side-by-side training pipelines for standard MLP Surrogate models and physically constrained PINNs (utilizing the Mish activation function).
- **Smart Early-Stopping:** A custom tracking protocol that monitors *Mean Physics Error (kN)* rather than just validation loss to capture the optimal structural weights (The "Best Epoch").
- **Real-Time Digital Twin UI:** A fully interactive Streamlit dashboard equipped with an Explainable AI (XAI) insight engine for Bridge Information Modeling (BrIM).

---

## 📂 Repository Structure

```
├── Data_Generators/
│   ├── ShortSpan_#01_Generator_16.41M_V7.py
│   ├── ShortSpan_#02_Generator_Validation_0.46M_V7.py
│   ├── LongSpan_#01_Generator_0.84M_V6.py
│   └── LongSpan_#02_Generator_Validation_0.25M_V6.py
├── AI_Training/
│   ├── ShortSpan__AITraining.py
│   └── LongSpan__AITraining.py
├── Application/
│   └── Result_Vizulizer.py
└── README.md
```

---

## 🛠️ Step-by-Step Execution Guide

To run this pipeline locally, you will need Python 3.9+ and a CUDA-enabled NVIDIA GPU (highly recommended for the massive training batches).

### Step 1: Clone & Install Dependencies

First, download the repository and install the required scientific libraries.

```bash
git clone https://github.com/YourUsername/Stress-Ribbon-Bridge-PINN.git
cd Stress-Ribbon-Bridge-PINN

# Install required dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy scikit-learn onnxruntime streamlit plotly
```


### Step 2: Data Generation (The Math) 🧮

Before training the AI, we need to teach it the laws of physics. The generator scripts sweep through millions of combinations of Spans, Height Differences, Distributed Loads, and Target Sags. It mathematically filters out physically impossible geometries and saves the viable ones to CSV datasets.

Run the generators for your desired domain:

```bash
# For Short Spans (Up to 300m)
python "Data_Generators/ShortSpan_#01_Generator_16.41M_V7.py"
python "Data_Generators/ShortSpan_#02_Generator_Validation_0.46M_V7.py"

# For Long Spans (Up to 2000m)
python "Data_Generators/LongSpan_#01_Generator_0.84M_V6.py"
python "Data_Generators/LongSpan_#02_Generator_Validation_0.25M_V6.py"
```

### Step 3: AI Training (The Brain) 🧠

Now we train the neural networks. The training scripts process the generated CSVs, normalize the data using Min-Max scaling, and train both a Surrogate Model and a PINN simultaneously.

> **Note:** The script automatically caps your GPU VRAM usage at 80% via CUDA to prevent Out-Of-Memory (OOM) crashes during massive batch loading.

```bash
# Train the Short Span Models
python "AI_Training/ShortSpan__AITraining.py"

# Train the Long Span Models
python "AI_Training/LongSpan__AITraining.py"
```

**Output:** The training will run for up to 500 epochs. The script will automatically isolate the "Best Epoch" (where physical fidelity peaks) and export the final weights as highly optimized `.onnx` files, alongside a `.json` Translation Dictionary for downstream use.
<img width="1892" height="908" alt="Generator_Validation_0 46M_V7_AIProcess" src="https://github.com/user-attachments/assets/a0244900-c621-4005-a21b-a82ab8341083" />


### Step 4: Interactive Visualizer (The Application) 🪄

Once the models are trained and saved as `.onnx` files, you can launch the interactive Streamlit web application. This acts as a real-time BrIM dashboard.

```bash
# Launch the interactive dashboard
streamlit run "Application/Result_Vizulizer.py"
```

Inside the Visualizer you can:

- Compare **Exact Math vs. Surrogate AI vs. PINN AI** side-by-side across 18 distinct structural parameters.
- View a live, dynamically updated plot of your cable catenary profile.
- Read automated **Explainable AI (XAI)** insights that warn you about capacity limits, span sensitivities, and geometric shifts.

- <img width="3783" height="1996" alt="Screenshot 2026-05-08 133134" src="https://github.com/user-attachments/assets/fd9da6b1-e456-4eb5-a54a-54aea3030f87" />
<img width="3687" height="1427" alt="Screenshot 2026-05-08 133155" src="https://github.com/user-attachments/assets/0a5302d2-e6ff-4055-b797-ebcb6008eb41" />



---

## 📜 Citation & Credits

This repository contains the code related to the research paper:

> *"Physics-Informed and Data-Driven Neural Networks for Real-Time Catenary Analysis in Cable-Supported Structures"*

**Credit:** Vijaykumar Parmar & Dr. K.B. Parikh
