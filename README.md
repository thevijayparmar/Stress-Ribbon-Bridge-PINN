\# 🌉 Explainable AI for Stress-Ribbon Bridges: PINN vs. Surrogate



!\[Python](https://img.shields.io/badge/Python-3.9+-blue.svg)

!\[PyTorch](https://img.shields.io/badge/PyTorch-CUDA\_Optimized-EE4C2C.svg)

!\[Streamlit](https://img.shields.io/badge/Streamlit-Interactive\_App-FF4B4B.svg)

!\[Status](https://img.shields.io/badge/Status-Research\_Ready-success.svg)



Welcome to the computational frontier of structural engineering! 🏗️ 



This repository contains the complete, end-to-end Machine Learning pipeline for designing and analyzing \*\*Cable-Supported \& Stress-Ribbon Bridges\*\*. By replacing traditional, mathematically expensive iterative solvers with \*\*Physics-Informed Neural Networks (PINNs)\*\* and \*\*Data-Driven Surrogate Models\*\*, this toolset achieves up to a \*\*33x computational speedup\*\* with sub-0.5% error.



Whether you're dealing with a pedestrian Short Span (< 300m) or a massive Long Span (< 2000m), this repo has the math, the models, and the UI to handle it.



\---



\## ✨ Features

\* \*\*Massive Data Generation:\*\* Exact-math catenary solvers capable of generating over 17 million physically viable structural combinations.

\* \*\*Dual AI Architectures:\*\* Side-by-side training pipelines for standard MLP Surrogate models and physically constrained PINNs (utilizing the Mish activation function).

\* \*\*Smart Early-Stopping:\*\* A custom tracking protocol that monitors \*Mean Physics Error (kN)\* rather than just validation loss to capture the optimal structural weights.

\* \*\*Real-Time Digital Twin UI:\*\* A fully interactive Streamlit dashboard equipped with an Explainable AI (XAI) insight engine.



\---



\## 🛠️ Installation \& Prerequisites



To run this pipeline locally, you'll need Python 3.9+ and a CUDA-enabled NVIDIA GPU (highly recommended for the massive training batches).



```bash

\# Clone the repository

git clone \[https://github.com/YourUsername/Stress-Ribbon-AI.git](https://github.com/YourUsername/Stress-Ribbon-AI.git)

cd Stress-Ribbon-AI



\# Install required dependencies

pip install torch torchvision torchaudio --index-url \[https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

pip install pandas numpy scikit-learn onnxruntime streamlit plotly

\## Credit
Credit: Vijaukumar Parmar & Dr. K.B. Parikh