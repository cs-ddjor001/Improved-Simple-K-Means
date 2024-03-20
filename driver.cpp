#include "k-means.h"

using namespace std;

int main() 
{
    vector<Point> points = {Point(1.0, 2.0), Point(2.0, 3.0), Point(3.0, 4.0), Point(5.0, 6.0), Point(7.0, 8.0)};
    int k = 2;
    int maxIterations = 10;

    KMeans kmeans(points, k);
    vector<Point> centroids = kmeans.run(maxIterations);

    cout << "Centroids:" << endl;
    for (const auto& centroid : centroids) {
        cout << "(" << centroid.x << ", " << centroid.y << ")" << endl;
    }

    return 0;
}
