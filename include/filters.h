// filters.h
// Author: Alex Perkins
// Date: 5/27/26
// Purpose: declares filter functions

#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

int greyscale(cv::Mat &src, cv::Mat &dst);
int sepia(cv::Mat &src, cv::Mat &dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

int negative(cv::Mat &src, cv::Mat &dst);
int blurSurroundings(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces);
int emboss(cv::Mat &src, cv::Mat &dst);

#endif