#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <vector>
using namespace std;

// Initializing vectors
vector<vector<double>> initialize(int rows, int cols, double value) 
{
    vector<vector<double>> matrix(rows, vector<double>(cols, value));
    return matrix;
}

// Function to initialize parameters
void initialize_parameters(vector<int> layer_dims, unordered_map<string, vector<vector<double>>> &parameters) 
{
    int L = layer_dims.size();
    for (int l = 1; l < L; ++l) 
    {
        parameters["W" + to_string(l)] = initialize(layer_dims[l], layer_dims[l-1], 0.1);
        parameters["b" + to_string(l)] = initialize(layer_dims[l], 1, 0.0);
    }
}

vector<vector<double>> linear_forward(vector<vector<double>>&A_prev, vector<vector<double>>&W, vector<vector<double>>&b) 
{
    vector<vector<double>> Z(W.size(), vector<double>(A_prev[0].size(), 0.0));
    
    // Matrix multiplication W * A_prev
    for (int i = 0; i < W.size(); ++i) 
    {
        for (int j = 0; j < A_prev[0].size(); ++j) 
        {
            for (int k = 0; k < W[0].size(); ++k) 
            {
                Z[i][j] += W[i][k] * A_prev[k][j];
            }
        }
    }
    
    // Adding bias 
    for (int i = 0; i < Z.size(); ++i) 
    {
        for (int j = 0; j < Z[0].size(); ++j) 
        {
            Z[i][j] += b[i][0];  
        }
    }

    return Z;
}

pair<vector<vector<double>>, vector<vector<vector<double>>>> L_layer_forward(vector<vector<double>> A_prev, unordered_map<string, vector<vector<double>>> &parameters, int layers) 
{
    vector<vector<double>> A = A_prev;
    vector<vector<vector<double>>> cache;
    cache.push_back(A_prev);
    
    for (int l = 1; l < layers; ++l) 
    {
        A_prev = A;
        vector<vector<double>> W = parameters["W" + to_string(l)];
        vector<vector<double>> b = parameters["b" + to_string(l)];
        A = linear_forward(A_prev, W, b);
        cache.push_back(A);
    }

    return {A, cache};
}

void update_parameters(unordered_map<string, vector<vector<double>>> &parameters, const vector<vector<vector<double>>> &cache, const vector<vector<double>> &X, double error, int layer, double learning_rate = 0.001) 
{
    // Update current layer parameters
    vector<vector<double>> &W = parameters["W" + to_string(layer)];
    vector<vector<double>> &b = parameters["b" + to_string(layer)];
    const vector<vector<double>> &A_prev = (layer == 1) ? X : cache[layer - 1];
    
    for (int i = 0; i < W.size(); ++i) 
    {
        for (int j = 0; j < W[0].size(); ++j) 
        {
            W[i][j] += learning_rate * 2 * error * A_prev[j][0];
        }
        b[i][0] += learning_rate * 2 * error;
    }

    if (layer > 1) 
    {
        // Calculate the propagated error for the previous layer
        vector<double> propagated_error(W[0].size(), 0.0);
        for (int j = 0; j < W[0].size(); ++j) 
        {
            for (int i = 0; i < W.size(); ++i) 
            {
                propagated_error[j] += error * W[i][j];
            }
        }

        // Recur for the previous layer
        update_parameters(parameters, cache, X, propagated_error[0], layer - 1, learning_rate);
    }
}

int main()
{
    vector<vector<double>> Dataset;
    ifstream file("data.txt");
    string line;

    if (file.is_open()) 
    {
        while (getline(file, line)) 
        {
            stringstream ss(line);
            vector<double> row;
            double value;
            
            while (ss >> value) 
            {
                row.push_back(value);
            }
            Dataset.push_back(row);
        }
        file.close();
    } 
    else 
    {
        cout << "Unable to open the Dataset file" << endl;
        return 0;
    }
    
    vector<int> layer_dims = {3, 2, 2, 1};
    int layers = layer_dims.size();
    unordered_map<string, vector<vector<double>>> parameters;
    initialize_parameters(layer_dims, parameters);
    
    int epochs = 50;
    double loss_sum;
    
    for(int e = 1; e <= epochs; e++)
    {
        loss_sum = 0.0;
        
        for(int d = 0; d < Dataset.size(); d++)
        {
            vector<vector<double>> X = {{Dataset[d][0]}, {Dataset[d][1]}, {Dataset[d][2]}};
            double Y = Dataset[d][3];
            
            pair<vector<vector<double>>, vector<vector<vector<double>>>> results = L_layer_forward(X, parameters, layers); // pair{y_hat, cache}
            double error = Y - results.first[0][0];
            update_parameters(parameters, results.second, X, error, layers-1);
            loss_sum += (error * error); // MSE - Loss Function 
        }
        
        cout << "Epoch:" << e << ", Loss: " << (loss_sum / Dataset.size()) << endl;
    }
    
    return 0;
}
