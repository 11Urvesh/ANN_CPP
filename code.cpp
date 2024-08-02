#include <sstream>
#include <iostream>
#include <bits/stdc++.h>
using namespace std;

// Initializing vectors
vector<vector<double>> initialize(int rows, int cols, double value) 
{
    vector<vector<double>> matrix(rows, vector<double>(cols, value));
    return matrix;
}

// Function to initialize parameters
void initialize_parameters(vector<int> layer_dims,unordered_map<string, vector<vector<double>>> &parameters) 
{
    int L = layer_dims.size();

    for (int l = 1; l < L; ++l) {
        parameters["W" + to_string(l)] = initialize(layer_dims[l-1], layer_dims[l],0.1);
        parameters["b" + to_string(l)] = initialize(layer_dims[l], 1, 0.0);
    }
}

vector<vector<double>> linear_forward(vector<vector<double>>&A_prev, vector<vector<double>>&W, vector<vector<double>>&b) 
{
    // Transpose of W
    vector<vector<double>> W_T(W[0].size(), vector<double>(W.size()));
    
    for (int i = 0; i < W.size(); ++i) 
    {
        for (int j = 0; j < W[0].size(); ++j) 
        {
            W_T[j][i] = W[i][j];
        }
    }
    
    
    // dot product [Transpose(W).A_prev]
    vector<vector<double>> Z(W_T.size(), vector<double>(A_prev[0].size(), 0.0));
    
    for (int i = 0; i < W_T.size(); ++i) 
    {
        for (int j = 0; j < A_prev[0].size(); ++j) 
        {
            for (int k = 0; k < A_prev.size(); ++k) 
            {
                Z[i][j] += W_T[i][k] * A_prev[k][j];
            }
        }
    }
    
    // Addind Bias to the result 
    for (int i = 0; i < Z.size(); ++i) 
    {
        for (int j = 0; j < Z[0].size(); ++j) 
        {
            Z[i][j] += b[i][0];
        }
    }

    return Z;
}

pair<vector<vector<double>>,vector<vector<double>>> L_layer_forward(vector<vector<double>> A_prev, unordered_map<string, vector<vector<double>>> &parameters,int layers) 
{
    vector<vector<double>> A = A_prev;
    int L = layers;

    for (int l = 1; l < L; ++l) 
    {
        A_prev = A;
        vector<vector<double>> W = parameters["W" + to_string(l)];
        vector<vector<double>> b = parameters["b" + to_string(l)];
        A = linear_forward(A_prev, W, b);
    }

    return {A,A_prev};
}

void update_parameters(unordered_map<string, vector<vector<double>>> &parameters,vector<vector<double>> &A1,vector<vector<double>>&X,const double error)
{
    parameters["W2"][0][0] += (0.001 * 2 * (error) * A1[0][0]);
    parameters["W2"][1][0] += (0.001 * 2 * (error) * A1[1][0]);
    parameters["b2"][0][0] += (0.001 * 2 * (error));
    
    parameters["W1"][0][0] += (0.001 * 2 * (error) * parameters["W2"][0][0] * X[0][0]);
    parameters["W1"][0][1] += (0.001 * 2 * (error) * parameters["W2"][0][0] * X[1][0]);
    parameters["W1"][0][2] += (0.001 * 2 * (error) * parameters["W2"][0][0] * X[2][0]);
    parameters["b1"][0][0] += (0.001 * 2 * (error) * parameters["W2"][0][0]);
    
    parameters["W1"][1][0] += (0.001 * 2 * (error) * parameters["W2"][1][0] * X[0][0]);
    parameters["W1"][1][1] += (0.001 * 2 * (error) * parameters["W2"][1][0] * X[1][0]);
    parameters["W1"][1][2] += (0.001 * 2 * (error) * parameters["W2"][1][0] * X[2][0]);
    parameters["b1"][1][0] += (0.001 * 2 * (error) * parameters["W2"][1][0]);
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
    
    /*
    Dataset Explanation : 
    
    Number of Bedrooms: The first value in each sub-vector.
    Size (sq ft/1000): The second value in each sub-vector.
    Location: The third value in each sub-vector, encoded as Urban = 1, Suburban = 2, Rural = 3.
    Rent (INR/10000): The fourth value in each sub-vector.
    
    */
    
    vector<int> layer_dims = {3, 2, 1};
    int layers = layer_dims.size();
    unordered_map<string, vector<vector<double>>> parameters;
    initialize_parameters(layer_dims,parameters);
    
    int epochs = 20;
    double loss_sum;
    
    for(int e = 1; e <= epochs; e++)
    {
        loss_sum = 0.0;
        
        for(int d = 0; d < Dataset.size();d++)
        {
            vector<vector<double>> X = {{Dataset[d][0]},{Dataset[d][1]},{Dataset[d][2]}};
            double Y = Dataset[d][3];
            
            pair<vector<vector<double>>,vector<vector<double>>>results = L_layer_forward(X,parameters,layers); // pair{y_hat,A1}
            
            double error = Y - results.first[0][0];
            update_parameters(parameters,results.second,X,error);
            
            loss_sum += (error*error); // MSE - Loss Function 
        }
        
        cout<<"Epoch:"<<e<<", Loss: "<<(loss_sum/Dataset.size())<<endl;
    }
    
    return 0;
}




