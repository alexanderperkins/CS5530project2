// filters.cpp
// Author: Alex Perkins
// Date: 5/27/26
// Purpose: image filter functions

#include "filters.h"
#include <opencv2/opencv.hpp>

// greyscales image
int greyscale(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), src.type());
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            // equal average of RGB channels
            uchar grey = (pixel[0] + pixel[1] + pixel[2]) / 3;
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(grey, grey, grey);
        }
    }
    return 0;
}

// applies sepia filter
int sepia(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), src.type());
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            float b = pixel[0];
            float g = pixel[1];
            float r = pixel[2];
            // apply sepia coefficients using original RGB values
            int newB = (int)(0.272*r + 0.534*g + 0.131*b);
            int newG = (int)(0.349*r + 0.686*g + 0.168*b);
            int newR = (int)(0.393*r + 0.769*g + 0.189*b);
            // clamp values to 255
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(
                newB > 255 ? 255 : newB,
                newG > 255 ? 255 : newG,
                newR > 255 ? 255 : newR
            );
        }
    }
    return 0;
}

// apply 5x5 blur filter using at method
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    // gaussian kernel
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8,16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };
    int kernelSum = 100;
    src.copyTo(dst);
    for(int i = 2; i < src.rows - 2; i++) {
        for(int j = 2; j < src.cols - 2; j++) {
            int b = 0, g = 0, r = 0;
            for(int ki = -2; ki <= 2; ki++) {
                for(int kj = -2; kj <= 2; kj++) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(i+ki, j+kj);
                    int w = kernel[ki+2][kj+2];
                    b += pixel[0] * w;
                    g += pixel[1] * w;
                    r += pixel[2] * w;
                }
            }
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(b/kernelSum, g/kernelSum, r/kernelSum);
        }
    }
    return 0;
}

// apply 5x5 blur using separable 1x5 filters
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    // separable 1x5 gaussian kernel
    int kernel[5] = {1, 2, 4, 2, 1};
    int kernelSum = 10;
    cv::Mat tmp;
    src.copyTo(tmp);
    src.copyTo(dst);

    // horizontal pass
    for(int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *tmpRow = tmp.ptr<cv::Vec3b>(i);
        for(int j = 2; j < src.cols - 2; j++) {
            int b = 0, g = 0, r = 0;
            for(int k = -2; k <= 2; k++) {
                b += srcRow[j+k][0] * kernel[k+2];
                g += srcRow[j+k][1] * kernel[k+2];
                r += srcRow[j+k][2] * kernel[k+2];
            }
            tmpRow[j] = cv::Vec3b(b/kernelSum, g/kernelSum, r/kernelSum);
        }
    }

    // vertical pass
    for(int i = 2; i < src.rows - 2; i++) {
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        for(int j = 0; j < src.cols; j++) {
            int b = 0, g = 0, r = 0;
            for(int k = -2; k <= 2; k++) {
                cv::Vec3b *tmpRow = tmp.ptr<cv::Vec3b>(i+k);
                b += tmpRow[j][0] * kernel[k+2];
                g += tmpRow[j][1] * kernel[k+2];
                r += tmpRow[j][2] * kernel[k+2];
            }
            dstRow[j] = cv::Vec3b(b/kernelSum, g/kernelSum, r/kernelSum);
        }
    }
    return 0;
}

// applies 3x3 sobel X filter as separable 1x3 filters
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), CV_16SC3);
    cv::Mat tmp;
    tmp.create(src.size(), CV_16SC3);

    // horizontal pass [1 0 -1]
    for(int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *tmpRow = tmp.ptr<cv::Vec3s>(i);
        for(int j = 1; j < src.cols - 1; j++) {
            for(int c = 0; c < 3; c++) {
                tmpRow[j][c] = -srcRow[j-1][c] + srcRow[j+1][c];
            }
        }
    }

    // vertical pass [1 2 1]
    for(int i = 1; i < src.rows - 1; i++) {
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);
        for(int j = 0; j < src.cols; j++) {
            for(int c = 0; c < 3; c++) {
                dstRow[j][c] = (tmp.ptr<cv::Vec3s>(i-1)[j][c] +
                                2 * tmp.ptr<cv::Vec3s>(i)[j][c] +
                                tmp.ptr<cv::Vec3s>(i+1)[j][c]) / 4;
            }
        }
    }
    return 0;
}

// applies 3x3 sobel Y filter as separable 1x3 filters
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), CV_16SC3);
    cv::Mat tmp;
    tmp.create(src.size(), CV_16SC3);

    // vertical pass [1 0 -1]
    for(int i = 1; i < src.rows - 1; i++) {
        for(int j = 0; j < src.cols; j++) {
            cv::Vec3b *above = src.ptr<cv::Vec3b>(i-1);
            cv::Vec3b *below = src.ptr<cv::Vec3b>(i+1);
            cv::Vec3s *tmpRow = tmp.ptr<cv::Vec3s>(i);
            for(int c = 0; c < 3; c++) {
                tmpRow[j][c] = -above[j][c] + below[j][c];
            }
        }
    }

    // horizontal pass [1 2 1]
    for(int i = 0; i < src.rows; i++) {
        cv::Vec3s *tmpRow = tmp.ptr<cv::Vec3s>(i);
        cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);
        for(int j = 1; j < src.cols - 1; j++) {
            for(int c = 0; c < 3; c++) {
                dstRow[j][c] = (tmpRow[j-1][c] +
                                2 * tmpRow[j][c] +
                                tmpRow[j+1][c]) / 4;
            }
        }
    }
    return 0;
}

// generates gradient magnitude image from sobel X and Y
// sx: sobel X image, sy: sobel Y image, dst: output magnitude image
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    dst.create(sx.size(), CV_8UC3);
    for(int i = 0; i < sx.rows; i++) {
        cv::Vec3s *sxRow = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *syRow = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        for(int j = 0; j < sx.cols; j++) {
            for(int c = 0; c < 3; c++) {
                // euclidean distance magnitude
                float mag = sqrt((float)(sxRow[j][c] * sxRow[j][c]) +
                                 (float)(syRow[j][c] * syRow[j][c]));
                dstRow[j][c] = mag > 255 ? 255 : (uchar)mag;
            }
        }
    }
    return 0;
}

// blurs and quantizes image into fixed number of levels
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    cv::Mat blurred;
    blur5x5_2(src, blurred);
    dst.create(src.size(), src.type());
    int b = 255 / levels;
    for(int i = 0; i < blurred.rows; i++) {
        cv::Vec3b *srcRow = blurred.ptr<cv::Vec3b>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        for(int j = 0; j < blurred.cols; j++) {
            for(int c = 0; c < 3; c++) {
                int xt = srcRow[j][c] / b;
                int xf = xt * b;
                dstRow[j][c] = (uchar)xf;
            }
        }
    }
    return 0;
}

// creates negative image subtracting pixels from 255
int negative(cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), src.type());
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(
                255 - pixel[0],
                255 - pixel[1],
                255 - pixel[2]
            );
        }
    }
    return 0;
}

// blurs surroundings around face
int blurSurroundings(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces) {
    blur5x5_2(src, dst);
    for(int i = 0; i < faces.size(); i++) {
        cv::Rect face = faces[i];
        face.x = std::max(0, face.x);
        face.y = std::max(0, face.y);
        face.width = std::min(face.width, src.cols - face.x);
        face.height = std::min(face.height, src.rows - face.y);
        src(face).copyTo(dst(face));
    }
    return 0;
}

// apply emboss using sobel
int emboss(cv::Mat &src, cv::Mat &dst) {
    cv::Mat sx, sy;
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);
    dst.create(src.size(), src.type());
    for(int i = 0; i < src.rows; i++) {
        cv::Vec3s *sxRow = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *syRow = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
        for(int j = 0; j < src.cols; j++) {
            for(int c = 0; c < 3; c++) {
                // dot product with direction (0.7071, 0.7071)
                float val = 0.7071 * sxRow[j][c] + 0.7071 * syRow[j][c];
                val = val + 128;
                dstRow[j][c] = val > 255 ? 255 : (val < 0 ? 0 : (uchar)val);
            }
        }
    }
    return 0;
}