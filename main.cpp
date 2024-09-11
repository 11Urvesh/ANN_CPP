
#include <iostream>
using namespace std;
#include "Dataset.h"
#include "ANN.h"

int main()
{
    cout<<"******************* Welcome to ANN.cpp project *******************"<<endl;
    Dataset dataset;
    dataset.loadData();

    ANN model;
   
    int epochs = 20;
    double loss_sum;
    
    for(int e = 1; e <= epochs; e++)
    {
        loss_sum = 0.0;
        
        for(int d = 0; d < dataset.size(); d++)
        {
            vector<vector<double>> X;
            double Y;
            dataset.getx(X,d);
            dataset.gety(Y,d);
            
            pair<vector<vector<double>>, vector<vector<vector<double>>>> results = model.L_layer_forward(X); // pair{y_hat, cache}
            double error = Y - results.first[0][0];
            model.update_parameters(results.second, X, error, model.getLayers()-1);
            loss_sum += (error * error); // MSE - Loss Function 
        }
        
        cout << "Epoch:" << e << ", Loss: " << (loss_sum / dataset.size()) << endl;
    }
    
    return 0;
}