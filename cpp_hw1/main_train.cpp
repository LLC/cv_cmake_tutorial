#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 
#include <iostream> 
#include "knn_train.h"

using namespace std;

int main() {
    cv::Mat img = cv::imread("/app/data/digits.png", 0);
    // cv::Mat img = cv::imread("./data/digits.png", 0);
    TrainDigit* train_obj = new TrainDigit();
    train_obj->train_val_split(img);
    train_obj->train();
    train_obj->validation();
    train_obj->save_model("/app/knn_model.yaml");
    // train_obj->save_model("./knn_model.yaml");
    return 0;
}