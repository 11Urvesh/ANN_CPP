#include <iostream>
using namespace std;
#include "Dataset.h"
#include "ANN.h"

int main()
{
    cout<<endl<<"******************* Welcome to ANN.cpp project *******************"<<endl;
    try
    {
        Dataset dataset;
        Dataset train_data;
        Dataset test_data;
    
        dataset.loadData();
        dataset.splitData(train_data, test_data, 0.8);

        ANN model(dataset);

        // model.load("model_weights.bin");

        int epochs;
        cout<<endl<<"Enter the number of epochs: ";
        cin>>epochs;

        if(epochs <= 0) throw string("Epochs can't be less than one !");

        cout<<endl<<"***********Training the model***********"<<endl;
        
        model.train(train_data, epochs);

        cout<<endl<<"***********Testing the model***********"<<endl;

        model.test(test_data);

        // model.save("model_weights.bin");

        cout<<endl<<"******************* ANN.cpp project completed *******************"<<endl;

    }
    catch(const string e)
    {
        cout<<e<<endl;
    }
    
    return 0;
}