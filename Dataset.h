#ifndef DATASET_H
#define DATASET_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
using namespace std;

class Dataset
{
    private: 
        vector<vector<double>> data;
    public:
        void loadData();
        int getFeatureCount();
        int getEntries();
        void getx(vector<vector<double>> &X,int &row);
        void gety(double &Y,int &row);
        void splitData(Dataset &train_data, Dataset &test_data, double split_ratio);
        friend class ANN;

        // All methods are defined in Dataset.cpp (MVC Architecture)
};

#endif // DATASET_H