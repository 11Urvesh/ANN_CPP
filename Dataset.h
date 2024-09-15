#ifndef DATASET_H
#define DATASET_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Dataset
{
    private: 
        vector<vector<double>> data;
    public:
        void loadData();
        int size();
        void getx(vector<vector<double>> &X,int &row);
        void gety(double &Y,int &row);
};

#endif // DATASET_H