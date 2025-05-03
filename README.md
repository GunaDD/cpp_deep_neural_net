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
* **g++ ≥ 10** (C++17)
* GNU make
* Python 3 (optional, for helper scripts)

### Build

**How to prepare the dataset:**

Download the MNIST data and place it at `data/mnist`.
It should contain:

```text
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
train-images-idx3-ubyte
train-labels-idx1-ubyte
```

Then run `make data` from the `cpp_deep_neural_det` directory. 
You should expect to see (on the `data/mnist` directory) : 
```text
t10k-images.f32
t10k-labels.u8
train-images.f32
train-labels.u8
```


```bash
# Clone
$ git clone https://github.com/GunaDD/cpp_deep_neural_net.git
$ cd cpp_deep_neural_net
```

**How to train the model:**
```bash
make main
./bin/main
```

Before running the commands, you can tune the model parameters (e.g. num of layers, learning rate schedule, weight initializations, experiment name) from the `main.cpp` file 

**How to inference:**

```bash
make inference
./bin/inference
```

Before running the command, you can specify the experiment name and the epoch number of the weights you want to load for to be used during inference time.


Executables are placed in **`bin/`**.


---

## 📊 Results (MNIST)

| Model                     | Parameters | Test accuracy |
| ------------------        | ---------: | ------------: |
| full_trainset_200_epoch   | 26506      |    **92.91%** |

---


## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
