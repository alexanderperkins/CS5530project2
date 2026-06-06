// features.h
// Author: Alex Perkins
// Date: 6/1/26
// Purpose: declares feature and distance functions

#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>

// 1. baseline matching - 7x7 center square
int sqCenter(cv::Mat &src, std::vector<float> &features);

// 1. baseline matching - sum-of-squared-difference
float SSD(std::vector<float> &f1, std::vector<float> &f2);

// 2. histogram matching - color histogram
int colorHistogram(cv::Mat &src, std::vector<float> &features, int bins);

// 2. histogram matching - histogram intersection distance
float histIntersection(std::vector<float> &f1, std::vector<float> &f2);

// 3. multihistogram matching - whole image and center
int multiHistogram(cv::Mat &src, std::vector<float> &features, int bins);

// 4. texture and color - texture histogram
int textureHistogram(cv::Mat &src, std::vector<float> &features, int bins);

// 4. texture and color - combined distance equal weights
float combinedDistance(std::vector<float> &f1, std::vector<float> &f2, int bins);

// 5. deep network embeddings - SSD distance
float SSDDistance(std::vector<float> &f1, std::vector<float> &f2);

// 7. custom design (stairs) - magnitude and color
int customFeature(cv::Mat &src, std::vector<float> &features, int bins);

#endif