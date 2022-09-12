#include "knn_test.h"

using namespace std;
using namespace cv;
using namespace cv::ml;


void TestDigit::load_model(string model_path)
{
    std::cout << "[start] load model" << std::endl;
    kclassifier = KNearest::load(model_path);  // opencv 4.5.3
    // kclassifier = StatModel::load<KNearest>(model_path);
    std::cout << "[finish] load model" << std::endl;
}

float TestDigit::inference(Mat& img)
{
    const int K = 5;
    const int cropSize = 20;
    Mat resized_img;
    resize(img, resized_img, Size(cropSize, cropSize), INTER_LINEAR);
    float TestDataF[1][cropSize * cropSize];
    memset(TestDataF, 0, cropSize * cropSize);
    Mat resized_img_flatten = resized_img.reshape(0, 1);

    for (int k = 0; k < resized_img_flatten.cols; k++)
    {
      TestDataF[0][k] = (float)resized_img_flatten.data[k];
    }

    Mat TestData = Mat(1, cropSize * cropSize, CV_32FC1, TestDataF);
    float f;
    f = kclassifier->predict(TestData);  // opencv 4.5.3
    // f = kclassifier->findNearest(TestData, K, f);
    return f;
}