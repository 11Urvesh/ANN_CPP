#include "Dataset.h"

void Dataset :: loadData() 
{
    ifstream file("Pune_rent_final.csv");
    string line;

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

int Dataset :: size()
{
    return data.size();
}

void Dataset :: getx(vector<vector<double>> &X,int &row)
{
    for (int i = 0; i < data[row].size() - 1; ++i) X.push_back({data[row][i]});
}

void Dataset :: gety(double &Y,int &row)
{
    Y = data[row].back();
}