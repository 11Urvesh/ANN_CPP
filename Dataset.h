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
        void loadData() 
        {
            ifstream file("data.csv");
            string line;

            try 
            {
                if(file.is_open() == false) throw string("Unable to open the Dataset file"); 

                while (getline(file, line)) 
                {
                    stringstream ss(line);
                    vector<double> row;
                    string value;
                    
                    while (getline(ss, value, ',')) 
                    {
                        row.push_back(stod(value));
                    }
                    data.push_back(row);
                }
                file.close();
            } 
            catch(const string e)
            {
                cout <<e<< endl;
            }
        }
        int size()
        {
            return data.size();
        }
        void getx(vector<vector<double>> &X,int &row)
        {
            for (int i = 0; i < data[row].size() - 1; ++i) X.push_back({data[row][i]});
        }
        void gety(double &Y,int &row)
        {
            Y = data[row].back();
        }
};

#endif // DATASET_H