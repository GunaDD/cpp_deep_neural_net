#include <bits/stdc++.h>

using namespace std;

/* 

Shallow network

Backpropagation

Y = XW + b

y = xW + b

dL/dW
dL/db
dL/dx

*/

random_device rd;
mt19937 gen(rd());
uniform_int_distribution<double> dist(0.0, 1.0);

const int SZ=10;

struct Net {
    double H[3][SZ];
    double W[2][SZ][SZ];
    double B[2][SZ];

    double bW[3][SZ][SZ];

    /* initialize the neural network */
    void init() {
        for(int l = 0; l < 2; l++) {
            for(int i = 0; i < SZ; i++) {
                for(int k = 0; k < SZ; k++) {
                    W[l][i][k] = dist(gen);
                }
            }
        }
    }   

    void load(vector<int> x) {
        for(int i = 0; i < SZ; i++) {
            H[0][i] = x[i];
        }
    }

    void forward() {
        for(int l = 0; l < 2; l++) {    
            for(int i = 0; i < SZ; i++) {
                for(int k = 0; k < SZ; k++) {
                    H[l+1][i] += W[l][i][k] * H[l][k];
                }
                H[l+1][i] += B[l][i];
                H[l+1][i] = max(0.0, H[l+1][i]); // relu 
            }
        }
    }

    /* Loss = -sum log softmax_{} */

    double softmax(int y) {
        double sum = 0;
        for(int i = 0; i < SZ; i++) {
            sum += exp(H[2][i]);
        }
        return exp(H[2][y]) / sum;
    }   

    double multiclass_cross_entropy_loss(vector<int> y) {
        double sum = 0;
        for(int i = 0; i < SZ; i++) {
            sum += log(softmax(y[i]));
        }
        sum = -sum; 
        return sum;
    }

    /* 
    let z_1, z_2, ..., z_10 be the outputs of the neural net
    let y_1, y_2, ..., y_s be the correct answer of the input x_1, x_2, ..., x_s

    total loss = - sum i from 1 to s log(softmax(y_i))
    softmax(y_i) = e^{z_{y_i}} / sum k=0 to 9 e^{k}

    dLoss / d z_i = dLoss/dsoftmax(y_i) * dsoftmax(y_i)/dz_i

    dLoss/dsoftmax(y_i) = -1/softmax(y_i) 
    dsoftmax(y_i) / dz_i = 
        if y_i == i:
            (e^{z_{y_i}} * (sum k=0 to 9 e^{k}) - e^{z_{y_i}} * e^{z_i}) / (sum k=0 to 9 e^{k})^2
            = e^{z_{y_i}} ( (sum k=0 to 9 e^{k}) - e^{z_i} ) / (sum k=0 to 9 e^{k})^2
            = softmax(y_i) * ( (sum k=0 to 9 e^{k}) - e^{z_i} )
        else:
            (- e^{z_{y_i}} * e^{z_i}) / (sum k=0 to 9 e^{k})^2 = softmax(y_i) * (-e^{z_i})

    dLoss / d z_i = 
        if y_i == i:
            (-1/softmax(y_i)) * softmax(y_i) * ( (sum k=0 to 9 e^{k}) - e^{z_i} ) = e^{z_i} - (sum k=0 to 9 e^{k})
        else:
            (-1/softmax(y_i)) * softmax(y_i) * (-e^{z_i}) = e^{z_i}

        so its equal to e^{z_i} - (y_i == i) * (sum k=0 to 9 e^{k})
    */
    void backward() {



    }
} net;

int main() {
    net.init();
    net.load({1,2,3,4,5,6,7,8,9,10});
    net.forward();

    for(int i = 0; i < SZ; i++) {
        cout << net.H[2][i] << " ";
    }
    cout << endl;

    cout << fixed << setprecision(9) << net.multiclass_cross_entropy_loss({1, 2, 3, 4, 5, 6, 7, 8, 9, 9}) << endl;


}