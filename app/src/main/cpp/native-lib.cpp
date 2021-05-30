#include <jni.h>
#include <opencv2/opencv.hpp>
#include "RPPG.hpp"

#define DEFAULT_RPPG_ALGORITHM "pca" //"g"// 
#define DEFAULT_FACEDET_ALGORITHM "deep"
#define DEFAULT_RESCAN_FREQUENCY 1
#define DEFAULT_SAMPLING_FREQUENCY 1
#define DEFAULT_MIN_SIGNAL_SIZE 5
#define DEFAULT_MAX_SIGNAL_SIZE 5
#define DEFAULT_DOWNSAMPLE 1 // x means only every xth frame is used

#define LOW_BPM 42
#define HIGH_BPM 240
#define SEC_PER_MIN 60

#define HAAR_CLASSIFIER_PATH "haarcascade_frontalface_alt.xml"
// #define DNN_PROTO_PATH "opencv/deploy.prototxt"
// #define DNN_MODEL_PATH "opencv/res10_300x300_ssd_iter_140000.caffemodel"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <android/log.h>
#define APPNAME "native-lib_cpp_file_log!"

const char* proto;
const char* weights;
const char* haar_model;

int i = 0;

RPPG rppg;

using namespace cv;

rPPGAlgorithm to_rppgAlgorithm(string s) {
    rPPGAlgorithm result;
    if (s == "g") result = g;
    else if (s == "pca") result = pca;
    else if (s == "xminay") result = xminay;
    else {
        //std::cout << "Please specify valid rPPG algorithm (g, pca, xminay)!" << std::endl;
        //exit(0);
    }
    return result;
}

faceDetAlgorithm to_faceDetAlgorithm(string s) {
    faceDetAlgorithm result;
    if (s == "haar") result = haar;
    else if (s == "deep") result = deep;
    else {
        //std::cout << "Please specify valid face detection algorithm (haar, deep)!" << std::endl;
        //exit(0);
    }
    return result;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myheartbeat2_MainActivity_test(JNIEnv *env, jobject thiz,
                                                       jstring p, jstring w, jstring h
) {
    proto = env->GetStringUTFChars(p, NULL);
    weights = env->GetStringUTFChars(w, NULL);
    haar_model = env->GetStringUTFChars(h, NULL);

//    env->ReleaseStringUTFChars(p, proto);
//    env->ReleaseStringUTFChars(w, weights);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myheartbeat2_MainActivity_setRPPG(JNIEnv *env, jobject thiz,
                                                       jint w, jint h
) {
    rPPGAlgorithm rPPGAlg;
    rPPGAlg = to_rppgAlgorithm(DEFAULT_RPPG_ALGORITHM);

    //cout << "Using rPPG algorithm " << rPPGAlg << "." << endl;

// face detection algorithm setting
    faceDetAlgorithm faceDetAlg;
    faceDetAlg = to_faceDetAlgorithm(DEFAULT_FACEDET_ALGORITHM);

    //cout << "Using face detection algorithm " << faceDetAlg << "." << endl;

// rescanFrequency setting
    double rescanFrequency;
    rescanFrequency = DEFAULT_RESCAN_FREQUENCY; //(double) rfrequency; //

// samplingFrequency setting
    double samplingFrequency;
    samplingFrequency = DEFAULT_SAMPLING_FREQUENCY;//(double) sfrequency; //

    __android_log_print(ANDROID_LOG_VERBOSE, "native-lib", "rescanFrequency=%f samplingFrequency=%f", rescanFrequency, samplingFrequency);

// max signal size setting
    int maxSignalSize;
    maxSignalSize = DEFAULT_MAX_SIGNAL_SIZE;

// min signal size setting
    int minSignalSize;
    minSignalSize = DEFAULT_MIN_SIGNAL_SIZE;

    if (minSignalSize > maxSignalSize) {
//        std::cout << "Max signal size must be greater or equal min signal size!" << std::endl;
        //exit(0);
    }

//// Reading downsample setting
//    int downsample;
//    downsample = DEFAULT_DOWNSAMPLE;

// Configure logfile path
//    string LOG_PATH;

// Load video information
    const double TIME_BASE = 0.001;
    const int WIDTH = (int) w;
    const int HEIGHT = (int) h;
//const double FPS = cv::getFps(matInput, TIME_BASE);

    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "w : %d h : %d", WIDTH, HEIGHT);

// Print video information
    //cout << "SIZE: " << WIDTH << "x" << HEIGHT << endl;
//cout << "FPS: " << FPS << endl;
    //cout << "TIME BASE: " << TIME_BASE << endl;

// Set up rPPG
    rppg = RPPG();

    rppg.load(rPPGAlg, faceDetAlg,
              WIDTH, HEIGHT, TIME_BASE,
              samplingFrequency, rescanFrequency,
              minSignalSize, maxSignalSize,
              haar_model,
              proto, weights);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myheartbeat2_MainActivity_main(JNIEnv *env, jobject thiz,
                                                                   jlong mat_addr_input,
                                                                   jlong mat_addr_result,
                                                                   jlong c_time
) {
    //int cnt = cv::getTickCount();
    //int freq = cv::getTickFrequency();

    //long time_now = clock();
    //int time = std::chrono::duration_cast<std::chrono::seconds>(start.time_since_epoch()).count(); //freq : 113537666 -> 원래프로그램 // 여기 : 1000000000

    //auto val = std::chrono::system_clock::now().time_since_epoch();
    //int mi = std::chrono::duration_cast<std::chrono::hours>(val).count();

    //__android_log_print(ANDROID_LOG_VERBOSE, APPNAME, "add : %ld", mat_addr_input);
    __android_log_print(ANDROID_LOG_VERBOSE, "native-lib", "c_time:%ld", c_time);

    if (i % DEFAULT_DOWNSAMPLE == 0) {
        Mat &matInput = *(Mat *)mat_addr_input;
        Mat &matResult = *(Mat *)mat_addr_result;

        //cout << "START ALGORITHM" << endl;

        rotate(matInput, matInput, ROTATE_90_COUNTERCLOCKWISE);

        Mat frameRGB;
        cvtColor(matInput, frameRGB, COLOR_RGBA2RGB);

        Mat frameGray;
        cvtColor(frameRGB, frameGray, COLOR_RGB2GRAY);

        equalizeHist(frameGray, frameGray);

        rppg.processFrame(frameRGB, frameGray, c_time);

        cvtColor(frameRGB, matResult, COLOR_RGB2RGBA);

        rotate(matResult, matResult, ROTATE_90_CLOCKWISE);
        rotate(matResult, matResult, ROTATE_180);

    } else {
        __android_log_print(ANDROID_LOG_VERBOSE, "skipping", "SKIPPING FRAME TO DOWNSAMPLE! i=%d", i);
    }
   // rppg.processFrame(frameRGB, frameGray, my_sys_time);

    i++;
}

extern "C"
JNIEXPORT void JNICALL

Java_com_example_myheartbeat2_MainActivity_deep(JNIEnv *env, jobject thiz,
                                                       jlong mat_addr_input,
                                                       jlong mat_addr_result
) {
    Mat &matInput = *(Mat *)mat_addr_input;
    Mat &matResult = *(Mat *)mat_addr_result;

    Net dnnClassifier = readNetFromCaffe(proto, weights);
    Mat resize300;
    cv::resize(matInput, resize300, Size(300, 300));
    Mat blob = blobFromImage(resize300, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0));
    dnnClassifier.setInput(blob);
    Mat detection = dnnClassifier.forward();
}