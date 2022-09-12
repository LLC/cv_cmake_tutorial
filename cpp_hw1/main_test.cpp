#include "knn_test.h"
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 
#include <iostream> 

using namespace std;

int main(int argc, char *argv[]) {
    cv::Mat img = cv::imread(argv[1], 0);
    TestDigit* test_obj = new TestDigit();
    test_obj->load_model("/app/knn_model.yaml");
    // test_obj->load_model("./knn_model.yaml");
    float result = test_obj->inference(img);
    cout << "Predict Result:" << result << endl;
    return 0;
}