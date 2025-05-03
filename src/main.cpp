#include <bits/stdc++.h>
#include "neural_net.h"

struct Dataset {
    std::vector<vector<float>> images;  
    std::vector<uint8_t> labels;
    uint32_t n;
    constexpr static uint32_t rows = 28, cols = 28;
};

Dataset load(const std::string& img_bin, const std::string& lbl_bin, uint32_t n) {
    Dataset ds;
    ds.n = n;

    ds.images = vector<std::vector<float>>(n, vector<float>(Dataset::rows * Dataset::cols));

    std::ifstream x(img_bin, std::ios::binary);
    if (!x) throw std::runtime_error("cannot open " + img_bin);

    for (uint32_t i = 0; i < n; ++i) {
        x.read(reinterpret_cast<char*>(ds.images[i].data()), Dataset::rows * Dataset::cols * sizeof(float));
        if (!x) throw std::runtime_error("failed reading image #" + std::to_string(i));
    }

    std::ifstream y(lbl_bin, std::ios::binary);
    if (!y) throw std::runtime_error("cannot open " + lbl_bin);
    ds.labels.resize(n);
    y.read(reinterpret_cast<char*>(ds.labels.data()), n);
    if (!y) throw std::runtime_error("failed reading labels");

    return ds;
}

int main() {
    const string experiment_name = "full_trainset_200_epoch_with_lr_schedule";
    const int num_train = 60000;
    const int num_val = 10000;
    const int num_epochs = 200;
    const int batch_size = 10; /* ensure batch_size divides num_train */
    // const double learning_rate = 0.0001;

    namespace fs = std::filesystem;

    string path = "../params/";
    path += experiment_name;

    ofstream out(path, ios::out | ios::trunc);

    out << "num_train: " << num_train << endl;
    out << "num_val: " << num_val << endl;
    out << "num_epochs: " << num_epochs << endl;

    vector<int> layers = {28*28, 32, 32, 10};

    out << "layers:";
    for(auto l: layers) {
        out << l << " ";
    }
    out << endl;

    vector<pair<int,ld>> lr_schedule = {{5, 2e-4}, {20, 1e-4}, {50, 1e-5}, {200, 1e-6}}; 

    out << "learning rate schedule: ";
    for(auto [epoch, lr] :lr_schedule) {
        out << epoch << " " << lr << "\n";
    }
    out << endl;

    Dataset train = load("./data/mnist/train-images.f32", "./data/mnist/train-labels.u8", num_train);

    /* normalize */
    for(auto &x:train.images) {
        for(auto &y: x) {
            y /= 255.0;
        }
    }

    Net nn(layers, batch_size, experiment_name); // takes in num_layers as parameter

    out << "train.images.size = " << train.images.size() << endl;

    nn.train(train.images, train.labels, num_epochs, lr_schedule);

    Dataset val = load("../data/mnist/t10k-images.f32", "../data/mnist/t10k-labels.u8", num_val);

    for(auto &x:val.images) {
        for(auto &y: x) {
            y /= 255.0;
        }
    }

    out << "val.images.size = " << val.images.size() << endl;

    int correct = 0;
    for(int k = 0; k < num_val; k++) {
        int pred = nn.eval(val.images[k], val.labels[k]);
        int label = int(val.labels[k]);
        correct += pred == label;
    }

    out << "accuracy : " << (ld) correct / num_val * 100 << endl;
}