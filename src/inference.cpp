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
    const int batch_size = 10;
    const int num_val = 10000;
    vector<int> layers = {28*28, 32, 32, 10};

    Net nn(layers, batch_size, experiment_name); // takes in num_layers as parameter

    string path = "../weights/";
    path += experiment_name;
    path += "/";
    path += "epoch40"; /* specify the epoch number */

    nn.load_model(path);
    Dataset val = load("../data/mnist/t10k-images.f32", "../data/mnist/t10k-labels.u8", num_val);

    for(auto &x:val.images) {
        for(auto &y: x) {
            y /= 255.0;
        }
    }

    cout << "val.images.size = " << val.images.size() << endl;

    int correct = 0;
    for(int k = 0; k < num_val; k++) {
        int pred = nn.eval(val.images[k], val.labels[k]);
        int label = int(val.labels[k]);
        correct += pred == label;
    }

    cout << "accuracy : " << (ld) correct / num_val * 100 << endl;
}