# KeWoKe-learning

Pipeline to train a **keyword spotting model**. The model is **quantized** and intended for deployment on the **KeWoKe ESP32 platform**.

---

## 🚀 Features

- End-to-end pipeline for keyword spotting  
- Python-based training and preprocessing scripts  
- Model quantization for efficient deployment on ESP32  
- Compatible with KeWoKe embedded audio recognition system  

---

## 📁 Repository Structure
```
KeWoKe-learning/
├── src/ # Training scripts, model definitions
├── pyproject.toml # Python project configuration
├── .gitignore # Ignored files (build, virtualenv, caches)
└── README.md # This file
```

---

## ⚙️ Requirements

- Python 3.8+  
- Dependencies listed in `pyproject.toml`  
- Virtual environment recommended (`.venv`)  

---

## 🛠 Setup

1. Clone the repository:

```bash
git clone https://github.com/LucReveyron/KeWoKe-learning.git
cd KeWoKe-learning
```

2. Create a virtual environment:
```bash
pip install -r requirements.txt
```
3. Install dependencies 
``` bash
pip install -r requirements.txt
```

## 🎯 Training

Run the trainning training script:
```bash
cd src
python train.py
```
- Data loading and preprocessing handled in src/

- Model is trained, evaluated, and quantized automatically
  
## 🧠 Model Deployment

- The quantized model is compatible with ESP32

- Integrate with KeWoKe firmware for on-device keyword spotting

## 📌 Notes

- Ignore folders like src/tuner_results/, build/, or .venv/

- Ensure .gitignore is correctly applied before committing large outputs

