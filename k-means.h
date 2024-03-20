#ifndef K_MEANS_H
#define K_MEANS_H

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <limits>
#include <algorithm> // For std::accumulate
#include <random>

struct Point 
{
    double x;
    double y;

    Point() : x(0.0), y(0.0) {}
    Point(double _x, double _y) : x(_x), y(_y) {}
};


class KMeans 
{
private:
    std::vector<Point> points;
    std::vector<Point> centroids;
    std::vector<int> assignments;
    std::vector<std::vector<double>> distances; // Matrix to store distances between points and centroids
    std::mt19937 mt; // Mersenne Twister random number engine

    double calculateDistance(const Point& p1, const Point& p2);

    void assignPointsToCentroids();

    void updateCentroids();

public:
    KMeans(const std::vector<Point>& _points, int k) : points(_points), centroids(k), assignments(_points.size()), mt(std::random_device{}()) {}

    std::vector<Point> run(int maxIterations);
};

#endif
