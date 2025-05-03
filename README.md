# cpp\_deep\_neural\_net

> **Neural networks from scratch in modern C++17 — zero third‑party dependencies.**

---

## ✨ Features

* **Pure C++17** implementation — no Eigen, no BLAS, no external frameworks
* **Xavier & Kaiming** weight initialisation
* Mini‑batch **SGD** (with momentum) & **Adam** optimiser
* Runs the full **MNIST** pipeline end‑to‑end 
* Single‑command build with **Make**

> *Work‑in‑progress*: convolutional layers, residual connections & transformer blocks.

---

## 🗂️ Repository layout

```text
.
├── src/            # C++ source 
│   ├── neural_net.h
│   ├── inference.cpp
│   └── main.cpp
├── scripts/        # Helper Python scripts (dataset download)
├── params/         # Hyper‑parameter files
├── weights/        # Saved checkpoints 
├── logs/           # Training logs 
└── README.md
```

---

## 🚀 Quick start

### Prerequisites

* Linux or macOS
* **g++ ≥ 10** or **clang ≥ 12** (C++17)
* **CMake ≥ 3.15** *or* GNU **make**
* Python 3 (optional, for helper scripts)

### Build

```bash
# Clone
$ git clone https://github.com/GunaDD/cpp_deep_neural_net.git
$ cd cpp_deep_neural_net
$ make main
```

Before running `make main`, you can tune the model parameters (e.g. num of layers, learning rate schedule, weight initializations, experiment name) from the `main.cpp` file 

Executables are placed in **`bin/`**.

If you only want to inference from trained model weight, 

---

## 📊 Results (MNIST)

| Model              | Parameters | Test accuracy |
| ------------------ | ---------: | ------------: |
|                    |            |    ** ** |

---


## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
