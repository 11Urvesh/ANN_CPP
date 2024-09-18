#ifndef ANN_H
#define ANN_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include "Dataset.h"
using namespace std;

class ANN
{
    private:
        int layers;
        vector<int> layer_dims;
        unordered_map<string, vector<vector<double>>> parameters;
    public:
        ANN(Dataset dataset);
        int getLayers();
        vector<vector<double>> linear_forward(vector<vector<double>>&A_prev, vector<vector<double>>&W, vector<vector<double>>&b);
        pair<vector<vector<double>>, vector<vector<vector<double>>>> L_layer_forward(vector<vector<double>> A_prev);
        void update_parameters(vector<vector<vector<double>>> &cache,vector<vector<double>> &X, double error, int layer, double learning_rate = 0.001);
        void train(Dataset &train_data, int epochs);
        void test(Dataset &test_data);

        // All methods are defined in ANN.cpp (MVC Architecture)
};

#endif // ANN_H