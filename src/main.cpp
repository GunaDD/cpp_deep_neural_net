#include <bits/stdc++.h>

using namespace std;

const int SZ=784;

/* 
let z_1, z_2, ..., z_10 be the outputs of the neural net
let y_1, y_2, ..., y_s be the correct answer of the input x_1, x_2, ..., x_s

total loss = - sum i from 1 to s log(softmax(y_i))
softmax(y_i) = e^{z_{y_i}} / sum k=0 to 9 e^{k}

dLoss / d z_i = dLoss/dsoftmax(y_i) * dsoftmax(y_i)/dz_i

dLoss/dsoftmax(y_i) = -1/softmax(y_i) 
dsoftmax(y_i) / dz_i = 
    if y_i == i:
        (e^{z_{y_i}} * (sum k=0 to 9 e^{z_k}) - e^{z_{y_i}} * e^{z_i}) / (sum k=0 to 9 e^{k})^2
        = e^{z_{y_i}} ( (sum k=0 to 9 e^{z_k}) - e^{z_i} ) / (sum k=0 to 9 e^{k})^2
        = softmax(y_i) * ( (sum k=0 to 9 e^{z_k}) - e^{z_i} )
    else:
        (- e^{z_{y_i}} * e^{z_i}) / (sum k=0 to 9 e^{z_k})^2 = softmax(y_i) * (-e^{z_i})

dLoss / d z_i = 
    if y_i == i:
        (-1/softmax(y_i)) * softmax(y_i) * ( (sum k=0 to 9 e^{k}) - e^{z_i} ) = e^{z_i} - (sum k=0 to 9 e^{z_k})
    else:
        (-1/softmax(y_i)) * softmax(y_i) * (-e^{z_i}) = e^{z_i}

    so its equal to e^{z_i} - (y_i == i) * (sum k=0 to 9 e^{z_k})
*/


/*
Going from the l+1-th layer to the l-th layer
Let y be the output of the l+1-th layer and x be the output of the l-th layer

then y = xW + b with W,b being weights and biases

we have dL/dy, we want dL/dx, dL/dW and dL/db so that we can update the weights and biases 
for this layer and update the previous layer

We suppose that we are training in one batch currently

Then y = xW + b

[y_1 y_2 .... y_N ] = [x_1 x_2 ... x_N ]    [w_11 ..... w_1N
                                                w_21 ..... w_2N
                                                .....
                                                w_M1 ..... W_MN]

dL/dW   = (dL / dy) * (dy / dW)
        = [dL/dy_1 .... dL/dy_N ] (dy / dw_11)

then we know that dL/dW is of the form

[dL/w_11 ..... dL/w_1N
    dL/w_21 ..... dL/w_2N
    .
    .
    .
    dL/w_M1 ..... dL/w_MN]

    so then dL/w_11 is a constant, also
    dL/dw_11 = [dL/dy_1 .... dL/dy_N] * (dy/dw_11)
    both of these implies dy/dw_11 is a column vector

    now looking back at the [y_1 .... y_n] = [x_1 .... x_m] W equality
    we get that

    dL/dw_11 = [x_1 0 0 0 0 ... 0]^T
    
    then similarly
    dL/dw_12 = [0 x_1 0 0 0 ... 0]^T
    dL/dw_21 = [x_2 0 0 0 0 ... 0]^T

then by dL/dw_11 = [dL/dy_1 .... dy_N] (dy/dw_11)
it follows dL/dw_11 = (dL/dy_1) * x1
similarly dL/dw_12 = (dL/dy_2) * x1, dL/w_21 = (dL/y_1) * x2 .... 

so dL/dw equal to
[(dL/dy_1) * x1, ..... , (dL/dy_N) * x1
    (dL/dy_1) * x2, ..... 
    .
    .
    (dL/dy_1) * xm, ..... , (dL/dy_N) * xm]


Now we want to derive dL/db

dL/db   = (dL/dy) * (dy/db)
        = [dL/dy_1 .... dL/dy_N] * (dy/db)

implies (dy/db) column vector
dy/db = [dy_1/db .... dy_N/db]^T

back to y = xW + b -> [y_1 ... y_N] = [x_1 ... x_M] W + [b .... b] (N times)
we see that dy_1/db = dy_2/db = ... = dy_N/db = 1
so dL/db = dL/dy

Now we want to find dL/dx

by chain rule dL/dx = (dL/dy) * (dy/dx)
since dimension of dL/dy is 1 x N and dimension of 1 x M
then dy/dx has dimension N x M

by expanding [ y_1 .... y_N ] = [ x_1 .... x_M ] W + b
we see that dy_1/dx_1 = w_11, dy_1/dx_2 = w_21, dy_2/dx_1 = w_12
hence dy/dx = W^T
so dL/dx = dL/dy W^T

*/

struct Net {
    vector<vector<vector<double>>> W;
    vector<vector<double>> B,H;
    vector<vector<vector<double>>> dW;
    vector<vector<double>> dB, dH;
 
    int num_layers; // num of hidden layer + 1 (includes the output layer)

    Net(vector<int> layers) { /* vector representing number of hidden unit in each layer */
        this->num_layers = layers.size();

        /* size of H is 1 more than size of W and B */
        W = {}, B = {}, H = {};
        for(int i = 0; i < layers.size();i++) {
            if(i + 1 < layers.size()) {
                W.push_back(vector<vector<double>>(layers[i], vector<double>(layers[i+1])));
                B.push_back(vector<double>(layers[i+1]));
            }
            H.push_back(vector<double>(layers[i]));
        }

        dW = W, dB = B, dH = H;

        /* Initialize the weights with Kaiming He initialization */

        const double gain = sqrt(2);
        const double std = gain / sqrt(static_cast<double>(SZ));

        mt19937_64 rng{random_device{}()};
        normal_distribution<double> dist(0.0, std);

        for (auto &w0 : W) {
            for(auto &w1 : w0) {
                for(auto &w2 : w1) {
                    w2 = dist(rng);
                }
            }   
        }
    }

    void load(vector<float> x) {
        assert(H[0].size() == x.size());

        for(int i = 0; i < x.size(); i++) {
            H[0][i] = x[i];
        }
    }

    void forward() {
        for(int l = 0; l + 1 < num_layers; l++) {  
            assert(W[l].size() == H[l].size());
            assert(W[l].size() > 0);
            assert(W[l][0].size() == H[l+1].size());

            for(int i = 0; i < H[l+1].size(); i++) {
                for(int k = 0; k < H[l].size(); k++) {
                    H[l+1][i] += H[l][k] * W[l][k][i]; 
                }
                H[l+1][i] += B[l][i];
                H[l+1][i] = max(0.0, H[l+1][i]); // relu 
            }
        }
    }

    /* Loss = -sum log softmax_{} */

    vector<double> softmax(vector<double> out_layer) {
        assert(out_layer.size() == 10); /* assuming output layer has 10 units */

        double sum = 0;
        for(int i = 0; i < out_layer.size(); i++) {
            sum += exp(out_layer[i]);
        }

        vector<double> res;
        for(int i = 0; i < out_layer.size(); i++) {
            res.push_back(exp(out_layer[i]) / sum);
        } 
        return res;
    }   

    /* multiclass cross-entropy loss */
    double cost(uint8_t label) {
        vector<double> applied_softmax = softmax(H[num_layers - 1]);
        return -log(applied_softmax[label]);
    }

    void backward(uint8_t y) {
        // initalizing the first dL/dy

        assert(dH[num_layers - 1].size() == 10);
        for(int i = 0; i < 10; i++) {
            dH[num_layers - 1][i] = exp(H[num_layers - 1][i]);
            if (y == i) {
                for(int k = 0; k < 10; k++) {
                    dH[num_layers - 1][i] -= exp(H[num_layers - 1][k]);
                }
            }
        }
        
        for(int l = num_layers - 2; l >= 0; l--) {  
            /* y = H[l+1], x = H[l], W = W[l], b = B[l]
            dL/dy = bH[l+1], dL/dx = bH[l] */

            for(int i = 0; i < H[l].size(); i++) { // iterate over X's
                // update dL/dW
                for(int k = 0; k < H[l+1].size(); k++) { // iterate over Y's
                    dW[l][i][k] = dH[l+1][k] * H[l][i];
                }      
            }

            for(int k = 0; k < H[l+1].size(); k++) {
                // update dL/db = dL/dy
                dB[l][k] = dH[l+1][k];
            }

            // dL/dx = dL/dy * W^T
            // dimensions: 1 x M = (1 x N) * (N x M)
            // y = xW + b
            // clean this up later with matmul and transpose functions

            /*
            Also note how the activation function (relu) affects the gradient
            particularly if xW + b < 0, then dL/dx = 0
            */

            for(int i = 0; i < H[l].size(); i++) { // iterate over X's

                double sum = 0;
                for(int k = 0; k < H[l+1].size(); k++) {
                    sum += H[l][i] * W[l][i][k] + B[l][i];
                }

                if (sum < 0) { // due to the ReLU
                    dH[l][i] = 0; 
                } else {
                    for(int k = 0; k < H[l+1].size(); k++) {
                        dH[l][i] += dH[l+1][k] * W[l][i][k]; // recall W[i,k] = W^T[k, i]
                    }
                }
            }

        }
    }

    void step(double learning_rate) {
        // update weights & biases
        for(int l = 0; l + 1 < num_layers; l++) {
            for(int i = 0; i < H[l].size(); i++) {
                B[l][i] -= learning_rate * dB[l][i];
                for(int k = 0; k < H[l+1].size(); k++) {
                    W[l][i][k] -= learning_rate * dW[l][i][k];
                }
            }
        }
    }

    void zero_grad() {
        // reset the gradients  
        for(int l = 0; l < num_layers; l++) {
            for(int i = 0; i < dH[l].size(); i++) {
                dH[l][i] = 0;
                H[l][i] = 0;
            }
        }

        for(int l = 0; l + 1 < num_layers; l++) {
            assert(dB[l].size() == H[l+1].size());
            for(int i = 0; i < H[l+1].size(); i++) {
                dB[l][i] = 0;
            }
            assert(dW[l].size() == H[l].size());
            assert(dW[l].size() > 0);
            assert(dW[l][0].size() == H[l+1].size());
            for(int i = 0; i < H[l].size(); i++) {
                for(int k = 0; k < H[l+1].size(); k++) {
                    dW[l][i][k] = 0;
                }
            }
        }
    }

    void train(vector<vector<float>> x, vector<uint8_t> y, int num_epochs, double learning_rate) {
        assert(x.size() == y.size());
        assert(x[0].size() == 28*28);

        for(int n = 0; n < num_epochs; n++) {
            /* one pass over all training data */
            double total_loss = 0;

            for(int m = 0; m < y.size(); m++) {
                load(x[m]);
                forward();
                total_loss += cost(y[m]);
                backward(y[m]);
                step(learning_rate);
                zero_grad();
            }

            cout << "epoch : " << n << " loss : " << total_loss << endl; 
        }   
    }

    int eval(vector<float> x, int label) {
        // cout << "receive " << label << '\n';
        load(x);
        forward();
        vector<double> applied_softmax = softmax(H[num_layers - 1]);
        assert(applied_softmax.size() == 10);

        int argmax = 0;
        for(int i = 0; i < 10; i++) {
            if(applied_softmax[argmax] < applied_softmax[i]) {
                argmax = i;
            }
        }
        return argmax;
    }
};

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
    const int num_samples = 10;
    const int num_epochs = 10;
    
    vector<int> layers = {28*28, 28*28, 10};

    Dataset ds = load("../data/mnist/train-images.f32", "../data/mnist/train-labels.u8", num_samples);

    /* normalize */
    for(auto &x:ds.images) {
        for(auto &y: x) {
            y /= 255.0;
        }
    }

    Net nn(layers); // takes in num_layers as parameter

    cout << "ds.images.size = " << ds.images.size() << " ds.labels.size = " << ds.labels.size() << endl;
    cout << "ds.images[0].size = " << ds.images[0].size() << endl;

    nn.train(ds.images, ds.labels, num_epochs, 0.001);

    for(int k = 0; k < num_samples; k++) {
        cout << "predicted: " << nn.eval(ds.images[k], ds.labels[k]) << " true label: " << int(ds.labels[k]) << endl;
    }

}