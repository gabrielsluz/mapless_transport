#include <iostream>
#include <vector> 
#include "polypartition.h"

using namespace std;

int main(){
    // Partition a simple polygon using polypartition.cpp
    //
    // Input: a simple polygon
    // Output: a set of simple polygons
    //
    // 1. Read the input polygon
    // 2. Partition the polygon
    // 3. Output the partitioned polygons
    //
    // Note: the input polygon is a simple polygon, which means that it is a
    //       closed polygon without self-intersection.
    //
    // Input format:
    // 1. The first line is the number of vertices of the polygon
    // 2. The following lines are the coordinates of the vertices
    //

    int N=0;
    vector<double> x,y;

    cin >> N;
    x.resize(N);
    y.resize(N);
    for(int i=0; i<N; i++){
        cin >> x[i] >> y[i];
    }

    // Partition using TPPLPartition::ConvexPartition_HM
    TPPLPoly poly;
    poly.Init(N);
    for(int i=0; i<N; i++){
        poly[i].x = x[i];
        poly[i].y = y[i];
    }

    TPPLPolyList partition;
    TPPLPartition pp;
    // int suc =  pp.ConvexPartition_HM(&poly, &partition);
    int suc =  pp.Triangulate_EC(&poly, &partition);
    
    //cout << suc << endl;

    // Output the partitioned polygons
    int num = partition.size();
    cout << "{\n\t\"polygons\": [\n";
    for(TPPLPolyList::iterator it = partition.begin(); it != partition.end(); it++){
        cout << "\t\t[";
        for(int i=0; i < it->GetNumPoints(); i++){
            cout << "[";
            cout << it->GetPoint(i).x << ", " << it->GetPoint(i).y;
            cout << "]";
            if(i != it->GetNumPoints()-1)
                cout << ",";
        }
        cout << "]";
        if(num-- != 1)
            cout << ",";
        cout << "\n";
    }
    cout << "\t]\n}";

    return 0;
}