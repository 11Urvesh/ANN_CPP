#include "ANN.h"

ANN :: ANN()
{
    cout<<"Enter the number of layers you want in ANN : ";
    cin>>layers;

    if(layers <= 1) throw string("Layers can't be less than two !");

    cout<<"*** Enter the number of neurons in each layer ***"<<endl;

    for(int i = 0; i < layers; i++)
    {
        int neurons;
        cout<<"Layer "<<i+1<<" : ";
        cin>>neurons;
        if(neurons <= 0) throw string("Neurons can't be less than zero !");
        layer_dims.push_back(neurons);
    }

    for (int l = 1; l < layers; ++l) 
    { 
        parameters["W" + to_string(l)] = vector<vector<double>>(layer_dims[l], vector<double>(layer_dims[l-1], 0.1)); 
        parameters["b" + to_string(l)] = vector<vector<double>>(layer_dims[l], vector<double>(1, 0.0));
    } 
}

int ANN :: getLayers()
{
    return layers;
}

vector<vector<double>> ANN :: linear_forward(vector<vector<double>>&A_prev, vector<vector<double>>&W, vector<vector<double>>&b) 
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

pair<vector<vector<double>>, vector<vector<vector<double>>>> ANN :: L_layer_forward(vector<vector<double>> A_prev) 
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

void ANN :: update_parameters(vector<vector<vector<double>>> &cache,vector<vector<double>> &X, double error, int layer, double learning_rate) 
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
        update_parameters(cache, X, propagated_error[0], layer - 1, learning_rate);
    }
}