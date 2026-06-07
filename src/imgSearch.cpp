// imgSearch.cpp
// Author: Alex Perkins
// Date: 6/1/26
// Purpose: content based image retrieval

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include "features.h"
#include "csv_util.h"

int main(int argc, char *argv[]) {
    // argument check
    if(argc < 5) {
        printf("Usage: %s <target image> <database dir> <num matches> <method>\n", argv[0]);
        printf("Methods: baseline histogram multihistogram texture dnn custom\n");
        return(-1);
    }

    // store arguments
    char *targetFile = argv[1];
    char *dbDir = argv[2];
    int numMatches = atoi(argv[3]);
    char *method = argv[4];

    // histogram bins
    const int bins = 8;

    // store distances
    std::vector<std::pair<float, std::string>> distances;

    // 5. deep network embeddings - read from CSV
    if(strcmp(method, "dnn") == 0) {
        // read CSV file
        char csvFile[256];
        sprintf(csvFile, "%s/ResNet18_olym.csv", dbDir);
        std::vector<char*> filenames;
        std::vector<std::vector<float>> data;
        read_image_data_csv(csvFile, filenames, data, 0);

        // find target in CSV
        std::vector<float> targetFeatures;
        for(int i = 0; i < filenames.size(); i++) {
            if(strstr(filenames[i], targetFile) != NULL ||
               strstr(targetFile, filenames[i]) != NULL) {
                targetFeatures = data[i];
                break;
            }
        }

        if(targetFeatures.empty()) {
            printf("Target not found in CSV\n");
            return(-1);
        }

        // compute distances
        for(int i = 0; i < filenames.size(); i++) {
            if(strstr(filenames[i], targetFile) != NULL ||
               strstr(targetFile, filenames[i]) != NULL) {
                continue;
            }
            float dist = SSDDistance(targetFeatures, data[i]);
            std::string fullPath = std::string(dbDir) + "/" + std::string(filenames[i]);
            distances.push_back(std::make_pair(dist, fullPath));
        }
    }
    else {
        // read target image
        cv::Mat target = cv::imread(targetFile);
        if(target.empty()) {
            printf("Unable to read %s\n", targetFile);
            return(-1);
        }

        // get target features
        std::vector<float> targetFeatures;

        // 1. baseline matching - get features
        if(strcmp(method, "baseline") == 0) {
            sqCenter(target, targetFeatures);
        }
        // 2. histogram matching - get features
        else if(strcmp(method, "histogram") == 0) {
            colorHistogram(target, targetFeatures, bins);
        }
        // 3. multihistogram matching - get features
        else if(strcmp(method, "multihistogram") == 0) {
            multiHistogram(target, targetFeatures, bins);
        }
        // 4. texture and color - get features
        else if(strcmp(method, "texture") == 0) {
            textureHistogram(target, targetFeatures, bins);
        }
        // 7. custom design (stairs) - get features
        else if(strcmp(method, "custom") == 0) {
            customFeature(target, targetFeatures, bins);
        }
        else {
            printf("Unknown method: %s\n", method);
            return(-1);
        }

        // get database images
        std::vector<cv::String> imageFiles;
        cv::glob(std::string(dbDir) + "/*.jpg", imageFiles);

        // loop through database
        for(int i = 0; i < imageFiles.size(); i++) {
            std::string dbFile = std::string(imageFiles[i]);
            std::string targetStr = std::string(targetFile);
            std::string dbName = dbFile.substr(dbFile.find_last_of("/\\") + 1);
            std::string targetName = targetStr.substr(targetStr.find_last_of("/\\") + 1);
            if(dbName == targetName) {
                continue;
            }
            cv::Mat dbImage = cv::imread(imageFiles[i]);
            if(dbImage.empty()) {
                continue;
            }

            // get database features
            std::vector<float> dbFeatures;

            // 1. baseline matching - get features
            if(strcmp(method, "baseline") == 0) {
                sqCenter(dbImage, dbFeatures);
            }
            // 2. histogram matching - get features
            else if(strcmp(method, "histogram") == 0) {
                colorHistogram(dbImage, dbFeatures, bins);
            }
            // 3. multihistogram matching - get features
            else if(strcmp(method, "multihistogram") == 0) {
                multiHistogram(dbImage, dbFeatures, bins);
            }
            // 4. texture and color - get features
            else if(strcmp(method, "texture") == 0) {
                textureHistogram(dbImage, dbFeatures, bins);
            }
            // 7. custom design (stairs) - get features
            else if(strcmp(method, "custom") == 0) {
                customFeature(dbImage, dbFeatures, bins);
            }

            // compute distance
            float dist = 0.0;

            // 1. baseline matching - SSD distance
            if(strcmp(method, "baseline") == 0) {
                dist = SSD(targetFeatures, dbFeatures);
            }
            // 2. histogram matching - intersection distance
            else if(strcmp(method, "histogram") == 0) {
                dist = histIntersection(targetFeatures, dbFeatures);
            }
            // 3. multihistogram matching - weighted distance
            else if(strcmp(method, "multihistogram") == 0) {
                int histSize = bins * bins * bins;
                std::vector<float> whole1(targetFeatures.begin(),
                    targetFeatures.begin() + histSize);
                std::vector<float> whole2(dbFeatures.begin(),
                    dbFeatures.begin() + histSize);
                std::vector<float> center1(targetFeatures.begin() + histSize,
                    targetFeatures.end());
                std::vector<float> center2(dbFeatures.begin() + histSize,
                    dbFeatures.end());
                float wholeDist = histIntersection(whole1, whole2);
                float centerDist = histIntersection(center1, center2);
                dist = 0.5 * wholeDist + 0.5 * centerDist;
            }
            // 4. texture and color - combined distance
            else if(strcmp(method, "texture") == 0) {
                dist = combinedDistance(targetFeatures, dbFeatures, bins);
            }
            // 7. custom design (stairs) - SSD on edge pattern
            else if(strcmp(method, "custom") == 0) {
                dist = SSD(targetFeatures, dbFeatures);
            }
            distances.push_back(std::make_pair(dist,
                std::string(imageFiles[i])));        
        }
    }

    // sort ascending
    std::sort(distances.begin(), distances.end());

    // print top N
    printf("Top %d matches for %s using %s:\n",
        numMatches, targetFile, method);
    for(int i = 0; i < numMatches && i < distances.size(); i++) {
        printf("%d: %s (distance: %.4f)\n", i+1,
            distances[i].second.c_str(), distances[i].first);
    }

    // extension - display matching images
    cv::Mat target_display;
    if(strcmp(method, "dnn") == 0) {
        std::string fullPath = std::string(dbDir) + "/" + std::string(targetFile);
        target_display = cv::imread(fullPath);
    }
    else {
        target_display = cv::imread(targetFile);
    }
    if(!target_display.empty()) {
        cv::Mat resized;
        cv::resize(target_display, resized, cv::Size(200, 200));
        // window 1 - target image only
        cv::imshow("Target Image", resized);
    }

    // window 2 - top matches only
    cv::Mat matchDisplay;
    for(int i = 0; i < numMatches && i < distances.size(); i++) {
        cv::Mat match = cv::imread(distances[i].second);            if(!match.empty()) {
            cv::Mat resized;
            cv::resize(match, resized, cv::Size(200, 200));
            if(matchDisplay.empty()) {
                matchDisplay = resized;
            }
            else {
                cv::hconcat(matchDisplay, resized, matchDisplay);
            }
        }
    }

    if(!matchDisplay.empty()) {
        cv::imshow("Top Matches", matchDisplay);
    }
    cv::waitKey(0);

    return(0);
}