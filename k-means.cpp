#include "k-means.h"

double KMeans::calculateDistance(const Point& p1, const Point& p2)
{
    return sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2));
}

void KMeans::assignPointsToCentroids()
{
    for (size_t i = 0; i < points.size(); ++i) 
    {
        double minDist = std::numeric_limits<double>::max();
        size_t closestCentroid = 0;
        for (size_t j = 0; j < centroids.size(); ++j) 
        {
            double dist = distances[i][j];
            if (dist < minDist) 
            {
                minDist = dist;
                closestCentroid = j;
            }
        }
        assignments[i] = closestCentroid;
    }
}

void KMeans::updateCentroids()
{
    std::vector<int> clusterSizes(centroids.size(), 0);
    std::vector<Point> newCentroids(centroids.size(), {0.0, 0.0});

    for (size_t i = 0; i < points.size(); ++i) 
    {
        int cluster = assignments[i];
        newCentroids[cluster].x += points[i].x;
        newCentroids[cluster].y += points[i].y;
        clusterSizes[cluster]++;
    }

    for (size_t i = 0; i < centroids.size(); ++i) 
    {
        if (clusterSizes[i] > 0) 
        {
            centroids[i].x = newCentroids[i].x / clusterSizes[i];
            centroids[i].y = newCentroids[i].y / clusterSizes[i];
        }
    }
}

std::vector<Point> KMeans::run(int maxIterations)
{
    if (points.empty() || centroids.size() == 0 || centroids.size() > points.size() || maxIterations <= 0) 
    {
        std::cerr << "Invalid input parameters." << std::endl;
        return {};
    }

    // Random initialization using K-means++
    centroids[0] = points[rand() % points.size()];
    for (size_t i = 1; i < centroids.size(); ++i) 
    {
        std::vector<double> probabilities(points.size());
        double totalDistSquared = 0.0;
        for (size_t j = 0; j < points.size(); ++j) 
        {
            double minDistSquared = std::numeric_limits<double>::max();
            for (size_t c = 0; c < i; ++c) 
            {
                double distSquared = calculateDistance(points[j], centroids[c]);
                minDistSquared = std::min(minDistSquared, distSquared);
            }
            probabilities[j] = minDistSquared;
            totalDistSquared += minDistSquared;
        }

        std::uniform_real_distribution<double> distribution(0, totalDistSquared);
        double rnd = distribution(mt);
        for (size_t j = 0; j < points.size(); ++j) 
        {
            rnd -= probabilities[j];
            if (rnd <= 0) 
            {
                centroids[i] = points[j];
                break;
            }
        }
    }

    // Precompute distances between points and centroids
    distances.resize(points.size(), std::vector<double>(centroids.size()));
    for (size_t i = 0; i < points.size(); ++i) 
    {
        for (size_t j = 0; j < centroids.size(); ++j) 
        {
            distances[i][j] = calculateDistance(points[i], centroids[j]);
        }
    }

    std::vector<Point> prevCentroids;
    for (int iter = 0; iter < maxIterations; ++iter) 
    {
        prevCentroids = centroids;
        assignPointsToCentroids();
        updateCentroids();

        // Check for convergence
        bool converged = true;
        for (size_t i = 0; i < centroids.size(); ++i) 
        {
            if (calculateDistance(centroids[i], prevCentroids[i]) > 1e-5) 
            {
                converged = false;
                break;
            }
        }
        if (converged) 
        {
            break;
        }
    }
    return centroids;
}
