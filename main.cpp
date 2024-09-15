#include <iostream>
using namespace std;
#include "Dataset.h"
#include "ANN.h"

int main()
{
    cout<<"******************* Welcome to ANN.cpp project *******************"<<endl;
    try
    {
        Dataset dataset;
        dataset.loadData();

        ANN model;
        int count = 0;
        int epochs = 10;
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
                count++;
                
                pair<vector<vector<double>>, vector<vector<vector<double>>>> results = model.L_layer_forward(X); 
                double error = Y - results.first[0][0];
                model.update_parameters(results.second, X, error, model.getLayers()-1);
                loss_sum += (error * error); // MSE - Loss Function 
            }
            
            cout << "Epoch:" << e << ", Loss: " << (loss_sum / dataset.size()) << endl;
        }
    }
    catch(const string e)
    {
        cout<<e<<endl;
    }
    
    return 0;
}