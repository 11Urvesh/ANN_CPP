#include "ANN.h"
#include "Dataset.h"

ANN :: ANN(Dataset dataset)
{
    cout<<endl<<"Enter the number of layers you want in ANN : ";
    cin>>layers;

    if(layers <= 1) throw string("Layers can't be less than two !");

    cout<<endl<<"Enter the number of neurons in each layer :"<<endl;

    for(int i = 0; i < layers; i++)
    {
        int neurons;
        cout<<"Layer "<<i+1<<" : ";
        cin>>neurons;

        if(i == 0 && neurons != dataset.getFeatureCount()-1) 
            throw string("Number of neurons in the first layer should be equal to the number of features in the dataset: "+to_string(dataset.getFeatureCount()-1));
        
        if(neurons <= 0) throw string("Neurons can't be less than zero !");

        layer_dims.push_back(neurons);
    }

    for (int l = 1; l < layers; ++l) 
    { 
        parameters["W" + to_string(l)] = vector<vector<double>>(layer_dims[l], vector<double>(layer_dims[l-1], 0.1)); 
        parameters["b" + to_string(l)] = vector<vector<double>>(layer_dims[l], vector<double>(1, 0.0));
    } 
}

void ANN :: load(string filename)
{
    int rows, cols;
    ifstream inFile(filename, ios::binary);

    if (!inFile.is_open()) 
    {
        throw string("Error opening file for reading!");
    }

    for (int l = 1; l < layers; ++l) 
    { 
        inFile.read((char*)(&rows), sizeof(rows));
        inFile.read((char*)(&cols), sizeof(cols));

        for (auto& row : parameters["W" + to_string(l)]) 
        {
            inFile.read((char*)(row.data()), row.size() * sizeof(double));
        }

        inFile.read((char*)(&rows), sizeof(rows));
        inFile.read((char*)(&cols), sizeof(cols));

        for (auto& row : parameters["b" + to_string(l)]) 
        {
            inFile.read((char*)(row.data()), row.size() * sizeof(double));
        }
    } 

    cout << "Weights successfully loaded from " << filename << endl;

    inFile.close();
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

void ANN :: train(Dataset &train_data, int epochs)
{
    cout<<endl;
    
    double loss_sum;
    for(int e = 1; e <= epochs; e++)
    {
        loss_sum = 0.0;
        for(int d = 0; d < train_data.getEntries(); d++)
        {
            vector<vector<double>> X;
            double Y;
            train_data.getx(X,d);
            train_data.gety(Y,d);
            
            pair<vector<vector<double>>, vector<vector<vector<double>>>> results = L_layer_forward(X); 
            double error = Y - results.first[0][0];
            update_parameters(results.second, X, error, getLayers()-1);
            loss_sum += (error * error); // MSE - Loss Function 
        }
        
        cout << "Epoch:" << e << ", Training Loss: " << (loss_sum / train_data.getEntries()) << endl;
    }
}

void ANN :: test(Dataset &test_data)
{
    double test_loss_sum = 0.0;

        for(int d = 0; d < test_data.getEntries(); d++)
        {
            vector<vector<double>> X;
            double Y;
            test_data.getx(X,d);
            test_data.gety(Y,d);
            
            pair<vector<vector<double>>, vector<vector<vector<double>>>> results = L_layer_forward(X); 
            double error = Y - results.first[0][0];
            update_parameters(results.second, X, error, getLayers()-1);
            test_loss_sum += (error * error); // MSE - Loss Function 
        }
        
        cout <<endl<<"Test Loss: " << (test_loss_sum / test_data.getEntries()) << endl;
}

void ANN :: save(string filename)
{
    ofstream outFile(filename, ios::binary);

    if (!outFile.is_open()) {
        cerr << "Error opening file for writing!" << endl;
        return;
    }
   
    int rows,cols;
    
    for(int l = 1; l < layers; ++l)
    {
        
        rows = parameters["W" + to_string(l)].size();
        cols = parameters["W" + to_string(l)][0].size();
        
        outFile.write((const char*)(&rows), sizeof(rows));
        outFile.write((const char*)(&cols), sizeof(cols));

        for (const auto& row : parameters["W" + to_string(l)]) 
        {
            outFile.write((const char*)(row.data()), row.size() * sizeof(double));
        }

        rows = parameters["b" + to_string(l)].size();
        cols = parameters["b" + to_string(l)][0].size();
        
        outFile.write((const char*)(&rows), sizeof(rows));
        outFile.write((const char*)(&cols), sizeof(cols));

        for (const auto& row : parameters["b" + to_string(l)]) 
        {
            outFile.write((const char*)(row.data()), row.size() * sizeof(double));
        }
    }

    outFile.close();

    cout << "Weights successfully stored in " << filename << endl;
}