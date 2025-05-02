#include <bits/stdc++.h>

using namespace std;

using ld = long double;

const int SZ = 784;
const ld eps = 1e-15;

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

const ld beta = 0.9;

struct Net {
    /* We stack the bias on the weight and add another column to x = [x_1 ... x_m 1] */
    vector<vector<vector<ld>>> W, H;
    vector<vector<vector<ld>>> dW, dH, momentum;
 
    int num_layers, batch_size; // num of hidden layer + 1 (includes the output layer)
    vector<int> layers;

    void train_init() {
        /* size of H is 1 more than size of W and B */
        W = {};
        H = {};
        for(int i = 0; i < layers.size();i++) {
            if(i + 1 < layers.size()) {
                /* +1 comes for the bias */
                W.push_back(vector<vector<ld>>(layers[i] + 1, vector<ld>(layers[i+1]))); 
                dW.push_back(vector<vector<ld>>(layers[i] + 1, vector<ld>(layers[i+1])));
            }
            H.push_back(vector<vector<ld>>(batch_size));
            dH.push_back(vector<vector<ld>>(batch_size, vector<ld>(layers[i])));
        }

        momentum = W;

        /* Initialize weight with Kaiming Initialization */
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

        /* Initialize the weights with Xavier/Glorot initialization */

        // mt19937_64 rng{random_device{}()};
        // for (auto &w0 : W) {
        //     ld std = sqrt((ld) 6 / (ld)(w0.size() + w0[0].size()));
        //     uniform_real_distribution<ld> dist(-std, std);
        //     for(auto &w1 : w0) {
        //         for(auto &w2 : w1) {
        //             w2 = dist(rng);
        //         }
        //     }   
        // }
    }

    void eval_init() {
        H = {};
        for(int i = 0; i < layers.size();i++) {
            H.push_back(vector<vector<ld>>(1)); /* batch_size = 1*/
        }
    }

    Net(vector<int> layers, int K) { /* vector representing number of hidden unit in each layer, K = batch size */
        this->num_layers = layers.size();
        this->batch_size = K; 
        this->layers = layers;
        train_init();
    }

    void load(vector<vector<float>> X) {
        assert(X.size() == batch_size || X.size() == 1);
        assert(X[0].size() == 28*28);

        for(int i = 0; i < X.size(); i++) {
            assert(H[0][i].empty());
            for(int j = 0; j < X[i].size(); j++) {
                H[0][i].push_back(X[i][j]);
            }
        }
    }

    vector<vector<ld>> matmul(vector<vector<ld>> A, vector<vector<ld>> B) {
        assert(A.size() > 0 && A[0].size() == B.size());
        int n = A.size();
        int r = A[0].size();
        int m = B[0].size();
        vector<vector<ld>> res = vector<vector<ld>>(n, vector<ld>(m));
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                for(int k = 0; k < r; k++) {
                    res[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return res;
    }

    vector<vector<ld>> transpose(vector<vector<ld>> A) {
        int n = A.size();
        int m = A[0].size();
        vector<vector<ld>> res(m, vector<ld>(n));
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                res[i][j] = A[j][i];
            }   
        }
        return res;
    }

    void ReLU(vector<vector<ld>> &A) {
        for(auto &x:A) {
            for(auto &y:x) {
                y = max(y, (ld)0);
            }
        }
    }

    void forward(int batch_size) {
        for(int l = 0; l + 1 < num_layers; l++) {
            auto X = H[l];
            for(int b = 0; b < batch_size; b++) {
                X[b].push_back((ld)1);
            }
            if(W[l].size() != X[0].size()) {
                cout << "layer " << l << " " << W[l].size() << " " << X[0].size() << endl;
            }
            assert(W[l].size() == X[0].size());
            H[l+1] = matmul(X, W[l]); 
            ReLU(H[l+1]);
        }
    }

    /* Loss = -sum log softmax_{} */
    vector<ld> softmax(vector<ld> out_layer) {
        assert(out_layer.size() == 10); /* assuming output layer has 10 units */

        ld m = *max_element(out_layer.begin(), out_layer.end());
        ld sum = 0;
        for(int i = 0; i < out_layer.size(); i++) {
            sum += exp(out_layer[i] - m);
        }

        vector<ld> res;
        ld check_sum = 0;
        for(int i = 0; i < out_layer.size(); i++) {
            res.push_back(exp(out_layer[i] - m) / sum);
            check_sum += res.back();
        } 

        return res;
    }   

    /* multiclass cross-entropy loss, takes in a set of labels */
    ld cost(vector<uint8_t> y) {
        assert(y.size() == batch_size);
        ld sum = 0;
        for(int j = 0; j < batch_size; j++) {
            auto z = H[num_layers - 1][j];
            vector<ld> applied_softmax = softmax(z);
            sum += -log(max(applied_softmax[y[j]], eps)); /* take the max with epsilon to prevent log(0) */
        }
        return sum / static_cast<ld>(batch_size);
    }

    /*
    Summary:
    For batch size = 1
    * dL / dW = matmul(x^T, dL/dy)
    * dL / db = dL / dy
    * dL / dx = matmul(dL/dy, W^T)

    Extending to batch size > 1
    * dL / dW : the loss is the sum over all elements in the batch
    dL / dW = matmul(x_1^T, dL / dy) + ... + matmul(x_k^T, dL / dy)
            = matmul(X^T, dL / dY)

            the dL / dY = stack each dL / dy for each item in the batch
            dL / dY \in R^{B x N}

    * dL / db = [ \sum dL/dy_1, ...., \sum dL / dy_N] where \sum run over all batchs

    * dL / dX = matmul(dL / dY, W^T) 
    to see this 
    we notice that dL / dX is equal to
    [   [dL / dx (for batch 1)]
        [dL / dx (for batch 2)]
        .
        .
        .
        [dL / dx (for batch K)]
    ] = 

    [ 
        [dL / dy (for batch 1)]
        .
        .
        .
        [dL / dy (for batch K)]
    ] * W^T
    we notice now the W^T must be the same (by equating how we get each row of the LHS)
    */

    void backward(vector<uint8_t> y) {
        // initalizing the first dL/dy
        assert(y.size() == batch_size);

        for(int b = 0; b < batch_size; b++) {
            assert(dH[num_layers - 1][b].size() == 10);
            ld m = *max_element(H[num_layers - 1][b].begin(), H[num_layers - 1][b].end());
            ld sum_exp = 0;
            for(int i = 0; i < 10; i++) {
                sum_exp += exp(H[num_layers - 1][b][i] - m);
            }

            for(int i = 0; i < 10; i++) {
                ld p = exp(H[num_layers - 1][b][i] - m) / sum_exp;
                dH[num_layers - 1][b][i] = p - (i == y[b] ? 1.0 : 0.0);
            }
        }

        for(int l = num_layers - 2; l >= 0; l--) {  
            /* y = H[l+1], x = H[l], W = W[l], b = B[l]
            dL/dy = bH[l+1], dL/dx = bH[l] */

            auto X = H[l], Y = H[l+1]; // for readability

            #define dX dH[l]
            #define dY dH[l+1]

            dW[l] = matmul(transpose(X), dY);
            assert(dW[l].size() == layers[l]);

            /*
            Also note how the activation function (relu) affects the gradient
            particularly if xW + b < 0, then dL/dx = 0
            */

            vector<vector<ld>> totdYdX = vector<vector<ld>>(layers[l+1], vector<ld>(layers[l]));
            for(int b = 0; b < batch_size; b++) {
                /* we process each batch separately to not create confusion
                create dY/dX for ReLU functions */

                vector<vector<ld>> dYdX = transpose(W[l]);
                for(int i = 0; i < layers[l+1]; i++) {
                    dYdX[i].pop_back();
                }
                assert(dYdX.size() == layers[l+1]);
                assert(dYdX[0].size() == layers[l]);
                for(int i = 0; i < Y[b].size(); i++) {
                    if(Y[b][i] <= 0) {
                        for(int j = 0; j < X[b].size(); j++) {
                            dYdX[i][j] = 0; 
                        }
                    }
                }
                
                for(int i = 0; i < layers[l+1]; i++) {
                    for(int j = 0; j < layers[l]; j++) {
                        totdYdX[i][j] += dYdX[i][j];
                    }
                }
            }

            assert(dY.size() == batch_size);
            assert(dY[0].size() == layers[l+1]);\

            dX = matmul(dY, totdYdX);
        }
    }

    void step(ld learning_rate) {
        // update weights & biases
        for(int l = 0; l + 1 < num_layers; l++) {
            /*
            ld norm = 0;
            for(int i = 0; i < layers[l]; i++) {
                for(int k = 0; k < layers[l+1]; k++) {
                    norm += W[l][i][k] * W[l][i][k];
                }
            }
            norm = sqrt(norm);
            ld clip_tresh = 100.0;
            if(norm > clip_tresh) {
                cout << "norm: " << norm << endl;
                for(int i = 0; i < layers[l]; i++) {
                    for(int k = 0; k < layers[l+1]; k++) {
                        W[l][i][k] *= clip_tresh / norm;
                    }
                }
            }
            */

            for(int i = 0; i < layers[l]; i++) {
                for(int k = 0; k < layers[l+1]; k++) {
                    momentum[l][i][k] = beta * momentum[l][i][k] + (1 - beta) * dW[l][i][k];
                    W[l][i][k] -= learning_rate * momentum[l][i][k];
                }
            }
        }
    }

    void zero_grad() {
        // reset the gradients  
        H = {}, dW = {};
        for(int i = 0; i < layers.size();i++) {
            H.push_back(vector<vector<ld>>(batch_size));
            if(i + 1 < layers.size()) {
                /* +1 comes for the bias */
                dW.push_back(vector<vector<ld>>(layers[i] + 1, vector<ld>(layers[i+1]))); 
            }
        }
    }

    void train(vector<vector<float>> x, vector<uint8_t> y, int num_epochs, ld learning_rate) {
        assert(x.size() == y.size());
        assert(x[0].size() == 28*28);

        mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
        for(int n = 0; n < num_epochs; n++) {
            vector<int> idx;
            for(int i = 0; i < x.size(); i++) idx.push_back(i);
            shuffle(idx.begin(), idx.end(), rng);

            ld total_loss = 0;

            int cur = 0;
            while(cur < x.size()) {
                vector<int> minibatch_indices;
                for(int j = 0; j < batch_size && cur + j < x.size(); j++) {
                    minibatch_indices.push_back(idx[cur + j]);
                }
                vector<vector<float>> minibatch_X;
                vector<uint8_t> minibatch_Y;
                for(auto i : minibatch_indices) {
                    minibatch_X.push_back(x[i]);
                    minibatch_Y.push_back(y[i]);
                }
                load(minibatch_X);
                forward(batch_size);
                total_loss += cost(minibatch_Y);
                backward(minibatch_Y);
                step(learning_rate);
                zero_grad();
                cur += batch_size;
            }

            total_loss /= x.size();
            cout << "epoch : " << n << " loss : " << total_loss << endl; 
 
            /* stop early when loss is already small enough to prevent precision issue occuring */
            if (total_loss < 1e-6) {
                break;
            }
        }   
    }

    int eval(vector<float> x, int label) {
        eval_init();
        load({x}); /* take batch_size = 1 for evals */ 
        forward(1);
        assert(H[num_layers - 1][0].size() == 10);
        vector<ld> applied_softmax = softmax(H[num_layers - 1][0]); /* since batch_size = 1 */
        assert(applied_softmax.size() == 10);

        auto argmax = max_element(applied_softmax.begin(), applied_softmax.end()) - applied_softmax.begin();
        return argmax;
    }

    void debug() {
        for(int i = 0; i < H[1].size(); i++) {
            for(int j = 0; j < H[2].size(); j++) {
                cout << fixed << setprecision(10) << W[1][i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
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
    const int num_train = 10000;
    const int num_val = 10000;
    const int num_epochs = 100;
    const int batch_size = 10; /* ensure batch_size divides num_train */
    const double learning_rate = 0.0001;

    cout << "num_train: " << num_train << endl;
    cout << "num_val: " << num_val << endl;
    cout << "num_epochs: " << num_epochs << endl;
    cout << "learning_rate: " << learning_rate << endl;

    vector<int> layers = {28*28, 32, 32, 10};

    Dataset train = load("../data/mnist/train-images.f32", "../data/mnist/train-labels.u8", num_train);

    /* normalize */
    for(auto &x:train.images) {
        for(auto &y: x) {
            y /= 255.0;
        }
    }

    Net nn(layers, batch_size); // takes in num_layers as parameter

    cout << "train.images.size = " << train.images.size() << endl;

    nn.train(train.images, train.labels, num_epochs, learning_rate);

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
        // cout << "predicted: " << pred << " true label: " << label << endl;
    }

    cout << "accuracy : " << (ld) correct / num_val * 100 << endl;
}