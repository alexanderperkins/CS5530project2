// features.cpp
// Author: Alex Perkins
// Date: 6/1/26
// Purpose: feature and distance functions

#include "features.h"
#include "filters.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

// 1. baseline matching - 7x7 center square
int sqCenter(cv::Mat &src, std::vector<float> &features) {
    features.clear();
    // find image center
    int centerRow = src.rows / 2;
    int centerCol = src.cols / 2;
    // get 7x7 patch
    for(int i = centerRow - 3; i <= centerRow + 3; i++) {
        for(int j = centerCol - 3; j <= centerCol + 3; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            features.push_back((float)pixel[0]);
            features.push_back((float)pixel[1]);
            features.push_back((float)pixel[2]);
        }
    }
    return 0;
}

// 1. baseline matching - sum-of-squared-difference
float SSD(std::vector<float> &f1, std::vector<float> &f2) {
    float diffSum = 0.0;
    for(int i = 0; i < f1.size(); i++) {
        float diff = f1[i] - f2[i];
        diffSum += diff * diff;
    }
    return diffSum;
}

// 2. histogram matching - color histogram
int colorHistogram(cv::Mat &src, std::vector<float> &features, int bins) {
    features.clear();
    // initialize histogram
    std::vector<float> hist(bins * bins * bins, 0.0);
    int binSize = 256 / bins;
    // count pixels per bin
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            int bBin = pixel[0] / binSize;
            int gBin = pixel[1] / binSize;
            int rBin = pixel[2] / binSize;
            hist[rBin * bins * bins + gBin * bins + bBin]++;
        }
    }
    // normalize by pixel count
    float total = src.rows * src.cols;
    for(int i = 0; i < hist.size(); i++) {
        features.push_back(hist[i] / total);
    }
    return 0;
}

// 2. histogram matching - histogram intersection distance
float histIntersection(std::vector<float> &f1, std::vector<float> &f2) {
    float intersection = 0.0;
    for(int i = 0; i < f1.size(); i++) {
        intersection += std::min(f1[i], f2[i]);
    }
    // distance is 1 minus intersection
    return 1.0 - intersection;
}

// 3. multihistogram matching - whole image and center
int multiHistogram(cv::Mat &src, std::vector<float> &features, int bins) {
    features.clear();
    // whole image histogram
    std::vector<float> wholeHist;
    colorHistogram(src, wholeHist, bins);
    // center region histogram
    int rowStart = src.rows / 4;
    int rowEnd = src.rows * 3 / 4;
    int colStart = src.cols / 4;
    int colEnd = src.cols * 3 / 4;
    cv::Mat center = src(cv::Rect(colStart, rowStart,
                        colEnd - colStart, rowEnd - rowStart));
    std::vector<float> centerHist;
    colorHistogram(center, centerHist, bins);
    // combine histograms
    for(int i = 0; i < wholeHist.size(); i++) {
        features.push_back(wholeHist[i]);
    }
    for(int i = 0; i < centerHist.size(); i++) {
        features.push_back(centerHist[i]);
    }
    return 0;
}

// 4. texture and color - texture histogram
int textureHistogram(cv::Mat &src, std::vector<float> &features, int bins) {
    features.clear();
    // get color histogram
    std::vector<float> colorHist;
    colorHistogram(src, colorHist, bins);
    // convert to greyscale
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    // compute sobel using opencv
    cv::Mat sx, sy, mag;
    cv::Sobel(grey, sx, CV_32F, 1, 0);
    cv::Sobel(grey, sy, CV_32F, 0, 1);
    cv::magnitude(sx, sy, mag);
    // build magnitude histogram
    std::vector<float> magHist(bins, 0.0);
    float binSize = 256.0 / bins;
    for(int i = 0; i < mag.rows; i++) {
        float *magRow = mag.ptr<float>(i);
        for(int j = 0; j < mag.cols; j++) {
            int bin = (int)(magRow[j] / binSize);
            if(bin >= bins) bin = bins - 1;
            magHist[bin]++;
        }
    }
    // normalize magnitude histogram
    float total = mag.rows * mag.cols;
    for(int i = 0; i < magHist.size(); i++) {
        magHist[i] /= total;
    }
    // combine color and texture
    for(int i = 0; i < colorHist.size(); i++) {
        features.push_back(colorHist[i]);
    }
    for(int i = 0; i < magHist.size(); i++) {
        features.push_back(magHist[i]);
    }
    return 0;
}

// 4. texture and color - combined distance equal weights
float combinedDistance(std::vector<float> &f1, std::vector<float> &f2, int bins) {
    // split color and texture
    int colorSize = bins * bins * bins;
    std::vector<float> color1(f1.begin(), f1.begin() + colorSize);
    std::vector<float> color2(f2.begin(), f2.begin() + colorSize);
    std::vector<float> texture1(f1.begin() + colorSize, f1.end());
    std::vector<float> texture2(f2.begin() + colorSize, f2.end());
    // equal weighted distance
    float colorDist = histIntersection(color1, color2);
    float textureDist = histIntersection(texture1, texture2);
    return 0.5 * colorDist + 0.5 * textureDist;
}

// 5. deep network embeddings - SSD distance
float SSDDistance(std::vector<float> &f1, std::vector<float> &f2) {
    float diffSum = 0.0;
    for(int i = 0; i < f1.size(); i++) {
        float diff = f1[i] - f2[i];
        diffSum += diff * diff;
    }
    return diffSum;
}

// 7. custom design - horizontal edge density per row
int customFeature(cv::Mat &src, std::vector<float> &features, int bins) {
    features.clear();
    // greyscale
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    // sobel X and Y
    cv::Mat sx, sy;
    sx.create(grey.size(), CV_16SC1);
    sy.create(grey.size(), CV_16SC1);
    for(int i = 1; i < grey.rows - 1; i++) {
        for(int j = 1; j < grey.cols - 1; j++) {
            sx.at<short>(i,j) =
                -grey.at<uchar>(i-1,j-1) + grey.at<uchar>(i-1,j+1)
                -2*grey.at<uchar>(i,j-1) + 2*grey.at<uchar>(i,j+1)
                -grey.at<uchar>(i+1,j-1) + grey.at<uchar>(i+1,j+1);
            sy.at<short>(i,j) =
                grey.at<uchar>(i-1,j-1) + 2*grey.at<uchar>(i-1,j)
                + grey.at<uchar>(i-1,j+1)
                -grey.at<uchar>(i+1,j-1) - 2*grey.at<uchar>(i+1,j)
                - grey.at<uchar>(i+1,j+1);
        }
    }
    // sum horizontal edges per row
    for(int i = 0; i < grey.rows; i++) {
        float rowSum = 0.0;
        for(int j = 0; j < grey.cols; j++) {
            rowSum += std::abs((float)sy.at<short>(i,j));
        }
        // normalize by row width
        features.push_back(rowSum / grey.cols);
    }
    return 0;
}