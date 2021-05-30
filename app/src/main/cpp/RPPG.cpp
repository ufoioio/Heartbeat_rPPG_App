//
//  RPPG.cpp
//  Heartbeat
//
//  Created by Philipp Rouast on 7/07/2016.
//  Copyright © 2016 Philipp Roüast. All rights reserved.
//

#include "RPPG.hpp"
#include "opencv.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>

#include <android/log.h>
#include <jni.h>

using namespace cv;
using namespace dnn;
//using namespace std;

#define LOW_BPM 42
#define HIGH_BPM 240
#define REL_MIN_FACE_SIZE 0.4
#define SEC_PER_MIN 60
#define MAX_CORNERS 10
#define MIN_CORNERS 5
#define QUALITY_LEVEL 0.01
#define MIN_DISTANCE 25

int num_sig = 0;
bool my_isset = false;
double my_mean_5_bpm;
double my_std_5_bpm;
Rect nose_mask;
bool is_love = false;

bool RPPG::load(const rPPGAlgorithm rPPGAlg, const faceDetAlgorithm faceDetAlg,
                const int width, const int height, const double timeBase,
                const double samplingFrequency, const double rescanFrequency,
                const int minSignalSize, const int maxSignalSize,
                const string &haarPath,
                const string &dnnProtoPath, const string &dnnModelPath) {

    this->rPPGAlg = rPPGAlg;
    this->faceDetAlg = faceDetAlg;
    this->lastSamplingTime = 0;
    this->minFaceSize = Size(min(width, height) * REL_MIN_FACE_SIZE, min(width, height) * REL_MIN_FACE_SIZE);
    this->maxSignalSize = maxSignalSize;
    this->minSignalSize = minSignalSize;
    this->rescanFlag = false;
    this->rescanFrequency = rescanFrequency;
    this->samplingFrequency = samplingFrequency;
    this->timeBase = timeBase;

    // Load classifier
    switch (faceDetAlg) {
      case haar:
        haarClassifier.load(haarPath);
        break;
      case deep:
        dnnClassifier = readNetFromCaffe(dnnProtoPath, dnnModelPath);
        break;
    }
    return true;
}

void RPPG::processFrame(Mat &frameRGB, Mat &frameGray, jlong c_time) {

    // Set time
    this->time = (int) (c_time%1000000000);
    //__android_log_print(ANDROID_LOG_VERBOSE, "rppg", "facevalid : %s", faceValid ? "true" : "false");
    __android_log_print(ANDROID_LOG_VERBOSE, "RPPG processFrame", "time: %d", time);

    if (!faceValid) {

        //cout << "Not valid, finding a new face" << endl;

        lastScanTime = time;
        detectFace(frameRGB, frameGray);

    } else if ((time - lastScanTime) * timeBase >= 1/rescanFrequency) {

        //cout << "Valid, but rescanning face" << endl;
        __android_log_print(ANDROID_LOG_VERBOSE, "RPPG processFrame", "(time - lastScanTime) = %d", (time - lastScanTime));

        lastScanTime = time;
        detectFace(frameRGB, frameGray);
        rescanFlag = true;

    } else {

        //cout << "Tracking face" << endl;

        trackFace(frameGray);
    }

    if (faceValid) {
        // Update fps
        fps = getFps(t, timeBase);

        // Remove old values from raw signal buffer
        while (s.rows > fps * maxSignalSize) {
            push(s);
            push(t);
            push(re);
            __android_log_print(ANDROID_LOG_VERBOSE, "RPPG processFrame", "pushing s.rows=%d", s.rows);
        }

        assert(s.rows == t.rows && s.rows == re.rows);

        // New values
        Scalar means = mean(frameRGB, mask);
        // Add new values to raw signal buffer
        double values[] = {means(0), means(1), means(2)};
        s.push_back(Mat(1, 3, CV_64F, values));
        __android_log_print(ANDROID_LOG_VERBOSE, "RPPG processFrame", "push_back s.rows=%d", s.rows);

        t.push_back(time);

        // Save rescan flag
        re.push_back(rescanFlag);
        //__android_log_print(ANDROID_LOG_VERBOSE, "rppg", "t.rows : %d", t.rows);

        // Update fps
        fps = getFps(t, timeBase);

        // Update band spectrum limits
        low = (int)(s.rows * LOW_BPM / SEC_PER_MIN / fps);
        high = (int)(s.rows * HIGH_BPM / SEC_PER_MIN / fps) +1;

        // If valid signal is large enough: estimate
        if (s.rows >= fps * minSignalSize) {

            // Filtering
            switch (rPPGAlg) {
                case g:
                    extractSignal_g();
                    break;
                case pca:
                    extractSignal_pca();
                    break;
                case xminay:
                    extractSignal_xminay();
                    break;
            }

            // HR estimation
            estimateHeartrate();
        }

        draw(frameRGB);
    }

    rescanFlag = false;

    frameGray.copyTo(lastFrameGray);
}

void RPPG::detectFace(Mat &frameRGB, Mat &frameGray) {

    //cout << "Scanning for faces" << endl;
    vector<Rect> boxes = {};

        switch (faceDetAlg) {
      case haar:
        // Detect faces with Haar classifier
        haarClassifier.detectMultiScale(frameGray, boxes, 1.1, 2, CASCADE_SCALE_IMAGE, minFaceSize);
        break;
      case deep:
        // Detect faces with DNN
        Mat resize300 = Mat();
        cv::resize(frameRGB, resize300, Size(300, 300));
        Mat blob = blobFromImage(resize300, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0));
        dnnClassifier.setInput(blob);
        Mat detection = dnnClassifier.forward();
        //Mat detectionMat = Mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        float confidenceThreshold = 0.5;

        for (int i = 0; i < detectionMat.rows; i++) {
          float confidence = detectionMat.at<float>(i, 2);
          if (confidence > confidenceThreshold) {
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frameRGB.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frameRGB.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frameRGB.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frameRGB.rows);
            Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));
            boxes.push_back(object);
          }
        }
        break;
    }

    if (boxes.size() > 0) {

        //cout << "Found a face" << endl;

        setNearestBox(boxes);
        detectCorners(frameGray);
        updateROI();
        updateMask(frameGray);
        faceValid = true;

    } else {

        //cout << "Found no face" << endl;
        invalidateFace();
    }
}

void RPPG::setNearestBox(vector<Rect> boxes) {
    int index = 0;
    Point p = box.tl() - boxes.at(0).tl();
    int min = p.x * p.x + p.y * p.y;
    for (int i = 1; i < boxes.size(); i++) {
        p = box.tl() - boxes.at(i).tl();
        int d = p.x * p.x + p.y * p.y;
        if (d < min) {
            min = d;
            index = i;
        }
    }
    box = boxes.at(index);
}

void RPPG::detectCorners(Mat &frameGray) {

    // Define tracking region
    Mat trackingRegion = Mat::zeros(frameGray.rows, frameGray.cols, CV_8UC1);
    Point points[1][4];
    points[0][0] = Point(box.tl().x + 0.22 * box.width,
                         box.tl().y + 0.21 * box.height);
    points[0][1] = Point(box.tl().x + 0.78 * box.width,
                         box.tl().y + 0.21 * box.height);
    points[0][2] = Point(box.tl().x + 0.70 * box.width,
                         box.tl().y + 0.65 * box.height);
    points[0][3] = Point(box.tl().x + 0.30 * box.width,
                         box.tl().y + 0.65 * box.height);
    const Point *pts[1] = {points[0]};
    int npts[] = {4};
    fillPoly(trackingRegion, pts, npts, 1, WHITE);

    // Apply corner detection
    goodFeaturesToTrack(frameGray,
                        corners,
                        MAX_CORNERS,
                        QUALITY_LEVEL,
                        MIN_DISTANCE,
                        trackingRegion,
                        3,
                        false,
                        0.04);
}

void RPPG::trackFace(Mat &frameGray) {

    // Make sure enough corners are available
    if (corners.size() < MIN_CORNERS) {
        detectCorners(frameGray);
    }

    Contour2f corners_1;
    Contour2f corners_0;
    vector<uchar> cornersFound_1;
    vector<uchar> cornersFound_0;
    Mat err;

    // Track face features with Kanade-Lucas-Tomasi (KLT) algorithm
    calcOpticalFlowPyrLK(lastFrameGray, frameGray, corners, corners_1, cornersFound_1, err);

    // Backtrack once to make it more robust
    calcOpticalFlowPyrLK(frameGray, lastFrameGray, corners_1, corners_0, cornersFound_0, err);

    // Exclude no-good corners
    Contour2f corners_1v;
    Contour2f corners_0v;
    for (size_t j = 0; j < corners.size(); j++) {
        if (cornersFound_1[j] && cornersFound_0[j]
            && norm(corners[j]-corners_0[j]) < 2) {
            corners_0v.push_back(corners_0[j]);
            corners_1v.push_back(corners_1[j]);
        } else {
            //cout << "Mis!" << std::endl;
        }
    }

    if (corners_1v.size() >= MIN_CORNERS) {

        // Save updated features
        corners = corners_1v;

        // Estimate affine transform
        Mat transform = estimateRigidTransform(corners_0v, corners_1v, false);

        if (transform.total() > 0) {

            // Update box
            Contour2f boxCoords;
            boxCoords.push_back(box.tl());
            boxCoords.push_back(box.br());
            Contour2f transformedBoxCoords;

            cv::transform(boxCoords, transformedBoxCoords, transform);
            box = Rect(transformedBoxCoords[0], transformedBoxCoords[1]);

            // Update roi
            Contour2f roiCoords;
            roiCoords.push_back(roi.tl());
            roiCoords.push_back(roi.br());
            Contour2f transformedRoiCoords;
            cv::transform(roiCoords, transformedRoiCoords, transform);
            roi = Rect(transformedRoiCoords[0], transformedRoiCoords[1]);

            updateMask(frameGray);
        }

    } else {
        //cout << "Tracking failed! Not enough corners left." << endl;
        invalidateFace();
    }
}

void RPPG::updateROI() {
    this->roi = Rect(Point(box.tl().x + 0.12 * box.width, box.tl().y + 0.4 * box.height),
                     Point(box.tl().x + 0.8 * box.width, box.tl().y + 0.65 * box.height)); //roi middle face
//    this->roi = Rect(Point(box.tl().x + 0.3 * box.width, box.tl().y + 0.1 * box.height),
//                     Point(box.tl().x + 0.7 * box.width, box.tl().y + 0.25 * box.height));


            /* Rect(Point(box.tl().x + 0.3 * box.width, box.tl().y + 0.1 * box.height),
                     Point(box.tl().x + 0.7 * box.width, box.tl().y + 0.25 * box.height)); */
}

void RPPG::updateMask(Mat &frameGray) {

    //cout << "Update mask" << endl;

    mask = Mat::zeros(frameGray.size(), frameGray.type());
    rectangle(mask, this->roi, WHITE, FILLED);
    nose_mask = Rect(this->roi.x+this->roi.width*0.2, this->roi.y, this->roi.width*(1-2*0.2), this->roi.height);
    rectangle(mask, nose_mask, 0, FILLED);
}

void RPPG::invalidateFace() {

    s = Mat1d();
    s_f = Mat1d();
    t = Mat1d();
    re = Mat1b();
    powerSpectrum = Mat1d();
    faceValid = false;
}

void RPPG::extractSignal_g() {

    // Denoise
    Mat s_den = Mat(s.rows, 1, CV_64F);
    denoise(s.col(1), re, s_den);

    // Normalise
    normalization(s_den, s_den);

    // Detrend
    Mat s_det = Mat(s_den.rows, s_den.cols, CV_64F);
    detrend(s_den, s_det, fps);

    // Moving average
    Mat s_mav = Mat(s_det.rows, s_det.cols, CV_64F);
    movingAverage(s_det, s_mav, 3, fmax(floor(fps/6), 2));

    s_mav.copyTo(s_f);
}

void RPPG::extractSignal_pca() {

    // Denoise signals
    Mat s_den = Mat(s.rows, s.cols, CV_64F);
    denoise(s, re, s_den);

    // Normalize signals
    normalization(s_den, s_den);

    // Detrend
    Mat s_det = Mat(s.rows, s.cols, CV_64F);
    detrend(s_den, s_det, fps);

    // PCA to reduce dimensionality
    Mat s_pca = Mat(s.rows, 1, CV_32F);
    Mat pc = Mat(s.rows, s.cols, CV_32F);
    pcaComponent(s_det, s_pca, pc, low, high);

    // Moving average
    Mat s_mav = Mat(s.rows, 1, CV_32F);
    movingAverage(s_pca, s_mav, 3, fmax(floor(fps/6), 2));

    s_mav.copyTo(s_f);
}

void RPPG::extractSignal_xminay() {

    // Denoise signals
    Mat s_den = Mat(s.rows, s.cols, CV_64F);
    denoise(s, re, s_den);

    // Normalize raw signals
    Mat s_n = Mat(s_den.rows, s_den.cols, CV_64F);
    normalization(s_den, s_n);

    // Calculate X_s signal
    Mat x_s = Mat(s.rows, s.cols, CV_64F);
    addWeighted(s_n.col(0), 3, s_n.col(1), -2, 0, x_s);

    // Calculate Y_s signal
    Mat y_s = Mat(s.rows, s.cols, CV_64F);
    addWeighted(s_n.col(0), 1.5, s_n.col(1), 1, 0, y_s);
    addWeighted(y_s, 1, s_n.col(2), -1.5, 0, y_s);

    // Bandpass
    Mat x_f = Mat(s.rows, s.cols, CV_32F);
    bandpass(x_s, x_f, low, high);
    x_f.convertTo(x_f, CV_64F);
    Mat y_f = Mat(s.rows, s.cols, CV_32F);
    bandpass(y_s, y_f, low, high);
    y_f.convertTo(y_f, CV_64F);

    // Calculate alpha
    Scalar mean_x_f;
    Scalar stddev_x_f;
    meanStdDev(x_f, mean_x_f, stddev_x_f);
    Scalar mean_y_f;
    Scalar stddev_y_f;
    meanStdDev(y_f, mean_y_f, stddev_y_f);
    double alpha = stddev_x_f.val[0]/stddev_y_f.val[0];

    // Calculate signal
    Mat xminay = Mat(s.rows, 1, CV_64F);
    addWeighted(x_f, 1, y_f, -alpha, 0, xminay);

    // Moving average
    movingAverage(xminay, s_f, 3, fmax(floor(fps/6), 2));
}

void RPPG::estimateHeartrate() {

    powerSpectrum = cv::Mat(s_f.size(), CV_32F);
    timeToFrequency(s_f, powerSpectrum, true);

    // band mask
    const int total = s_f.rows;
    Mat bandMask = Mat::zeros(s_f.size(), CV_8U);
    bandMask.rowRange(min(low, total), min(high, total) + 1).setTo(ONE);

    __android_log_print(ANDROID_LOG_VERBOSE, "rppghigh", "low:%d, high:%d total:%d", low, high, total);

    if (!powerSpectrum.empty()) {

        // grab index of max power spectrum
        double min, max;
        Point pmin, pmax;
        minMaxLoc(powerSpectrum, &min, &max, &pmin, &pmax, bandMask);

        Mat tmp = cv::Mat(powerSpectrum.rows, 1, CV_32F);
        cv::sortIdx(powerSpectrum, tmp, SORT_EVERY_COLUMN + SORT_DESCENDING);

        __android_log_print(ANDROID_LOG_VERBOSE, "tmp", "tmp.at<int>(1, 0)=%d tmp.at<int>(0, 0)=%d", tmp.at<int>(1, 0), tmp.at<int>(0, 0));
        __android_log_print(ANDROID_LOG_VERBOSE, "tmp", "powerSpectrum.at<double>(1, 0)=%f", powerSpectrum.at<double>(tmp.at<int>(1, 0), 0));
        __android_log_print(ANDROID_LOG_VERBOSE, "tmp", "powerSpectrum.rows=%d", powerSpectrum.rows);
        __android_log_print(ANDROID_LOG_VERBOSE, "tmp", "powerSpectrum.cols=%d", powerSpectrum.cols);

        // calculate BPM
        bpm = pmax.y * fps / total * SEC_PER_MIN;
        bpms.push_back(bpm);

        __android_log_print(ANDROID_LOG_VERBOSE, "tmp", "pmax.x=%d, pmax.y=%d", pmax.x, pmax.y);

        //cout << "FPS=" << fps << " Vals=" << powerSpectrum.rows << " Peak=" << pmax.y << " BPM=" << bpm << endl;
    }

    if ((time - lastSamplingTime) * timeBase >= 1/samplingFrequency) {
        lastSamplingTime = time;

        cv::sort(bpms, bpms, SORT_EVERY_COLUMN);

        // average calculated BPMs since last sampling time
        meanBpm = mean(bpms)(0);
        minBpm = bpms.at<double>(0, 0);
        maxBpm = bpms.at<double>(bpms.rows-1, 0);

        //std::cout << "meanBPM=" << meanBpm << " minBpm=" << minBpm << " maxBpm=" << maxBpm << std::endl;
        my_bpms.push_back(meanBpm);
        bpms.pop_back(bpms.rows);

        if(my_bpms.rows == 6) {
            my_bpms = my_bpms.rowRange(1, 6);
            Mat tmp_mean;
            Mat tmp_std;
            meanStdDev(my_bpms, tmp_mean, tmp_std);

            __android_log_print(ANDROID_LOG_VERBOSE, "my_bpm",
                                "tmp_mean=%f, tmp_std=%f, my_isset=%d",
                                tmp_mean.at<double>(0), tmp_std.at<double>(0),
                                my_isset);

            if (!my_isset & (tmp_std.at<double>(0) < 1)) {
                my_mean_5_bpm = tmp_mean.at<double>(0);
                my_std_5_bpm = tmp_std.at<double>(0);

                my_isset = true;

                __android_log_print(ANDROID_LOG_VERBOSE, "my_bpm",
                                    "my_mean_5_bpm=%f, my_std_5_bpm=%f, my_isset=%d",
                                    my_mean_5_bpm, my_std_5_bpm,
                                    my_isset);
            }

            __android_log_print(ANDROID_LOG_VERBOSE, "my_bpm",
                                "my_bpms.at<double>(my_bpms.rows-1)=%f",
                                my_bpms.at<double>(my_bpms.rows-1));

            //double check_sig = (my_bpms.at<double>(my_bpms.rows-1)-my_mean_5_bpm)/my_std_5_bpm;
            double check_sig = (tmp_mean.at<double>(0)-my_mean_5_bpm)/sqrt(my_std_5_bpm*my_std_5_bpm/5 + tmp_std.at<double>(0)*tmp_std.at<double>(0)/5);


            if(my_isset & check_sig > 2.132){
                //num_sig += 1;
                is_love = true;
            }
            else if(my_isset & check_sig <= 2.132){
                is_love = false;
            }

            __android_log_print(ANDROID_LOG_VERBOSE, "my_bpm", "check_sig=%f num_sig=%d", check_sig, num_sig);
        }
    }
}

void RPPG::draw(cv::Mat &frameRGB) {

    // Draw roi
    rectangle(frameRGB, roi, GREEN);

    // Draw bounding box
    rectangle(frameRGB, box, BLUE);

    // Draw signal
    if (!s_f.empty() && !powerSpectrum.empty()) {

        // Display of signals with fixed dimensions
        double displayHeight = box.height/2.0;
        double displayWidth = box.width*0.8;

        // Draw signal
        double vmin, vmax;
        Point pmin, pmax;
        minMaxLoc(s_f, &vmin, &vmax, &pmin, &pmax);
        double heightMult = displayHeight/(vmax - vmin);
        double widthMult = displayWidth/(s_f.rows - 1);
        double drawAreaTlX = box.tl().x + box.width + 20;
        double drawAreaTlY = box.tl().y;
        Point p1(drawAreaTlX, drawAreaTlY + (vmax - s_f.at<double>(0, 0))*heightMult);
        Point p2;
        for (int i = 1; i < s_f.rows; i++) {
            p2 = Point(drawAreaTlX + i * widthMult, drawAreaTlY + (vmax - s_f.at<double>(i, 0))*heightMult);
            line(frameRGB, p1, p2, BLUE, 2);
            p1 = p2;
        }

        // Draw powerSpectrum
        const int total = s_f.rows;
        Mat bandMask = Mat::zeros(s_f.size(), CV_8U);
        bandMask.rowRange(min(low, total), min(high, total)+1).setTo(ONE);
        minMaxLoc(powerSpectrum, &vmin, &vmax, &pmin, &pmax, bandMask);
        heightMult = displayHeight/(vmax - vmin);
        widthMult = displayWidth/(high - low);
        drawAreaTlX = box.tl().x + box.width + 20;
        drawAreaTlY = box.tl().y + box.height/2.0;
        p1 = Point(drawAreaTlX, drawAreaTlY + (vmax - powerSpectrum.at<double>(low, 0))*heightMult);
        for (int i = low + 1; i <= high; i++) {
            p2 = Point(drawAreaTlX + (i - low) * widthMult, drawAreaTlY + (vmax - powerSpectrum.at<double>(i, 0)) * heightMult);
            line(frameRGB, p1, p2, BLUE, 2);
            p1 = p2;
        }
    }

    std::stringstream ss;

    // Draw BPM text
    if (faceValid) {
        ss.precision(3);
        ss << meanBpm << " bpm";
        putText(frameRGB, ss.str(), Point(box.tl().x, box.tl().y - 10), FONT_HERSHEY_PLAIN, 6, BLUE, 4);
    }

    // Draw FPS text
    ss.str("");
    ss << fps << " fps";
    putText(frameRGB, ss.str(), Point(box.tl().x, box.br().y + 40), FONT_HERSHEY_PLAIN, 6, GREEN, 4);

    // Draw corners
    for (int i = 0; i < corners.size(); i++) {
        //circle(frameRGB, corners[i], r, WHITE, -1, 8, 0);
        line(frameRGB, Point(corners[i].x-5,corners[i].y), Point(corners[i].x+5,corners[i].y), GREEN, 3);
        line(frameRGB, Point(corners[i].x,corners[i].y-5), Point(corners[i].x,corners[i].y+5), GREEN, 3);
    }

    //draw my bpms
    putText(frameRGB, to_string(my_mean_5_bpm), Point(box.br().x, box.br().y + 50), FONT_HERSHEY_PLAIN, 3, GREEN, 4);

    if(is_love){
        putText(frameRGB, "love", Point(box.br().x, box.br().y + 10), FONT_HERSHEY_PLAIN, 6, RED, 4);
    }
    else{
        putText(frameRGB, "not love", Point(box.br().x, box.br().y + 10), FONT_HERSHEY_PLAIN, 6, BLUE, 4);
    }

    rectangle(frameRGB, nose_mask, BLACK, FILLED);
}
