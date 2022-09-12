#include "knn_train.h"


using namespace std;
using namespace cv;
using namespace cv::ml;


void TrainDigit::train_val_split(Mat& img)
{
    // 圖片共有 10 類
    const int classSum = 10;
    int imgW = img.cols;
    int imgH  = img.rows;
    const int cropSize = 20;
    int WSplit = imgW / cropSize;
    int HSplit = imgH / cropSize;

    std::cout << "圖的寬為" << imgW << ",原圖的高為" << imgH << std::endl;
    std::cout << "切圖的寬為" << cropSize << ",切圖的高為" << cropSize << std::endl;
    std::cout << "水平一共會切" << WSplit << "個小圖，垂直一共會切" << HSplit << "個小圖" << std::endl;

    int cropTotal = imgW / cropSize * imgH / cropSize;
    std::cout << "總共會有" << cropTotal << "個小圖" << std::endl;
    int trainTotal = cropTotal / 2;

    float TrainDataF[trainTotal][cropSize * cropSize];
    memset(TrainDataF, 0, trainTotal * cropSize * cropSize);
    float TrainLabelF[trainTotal];

    float ValDataF[trainTotal][cropSize * cropSize];
    memset(ValDataF, 0, trainTotal * cropSize * cropSize);
    float ValLabelF[trainTotal];

    // 將原圖切成小圖
    for(int i = 0; i < WSplit; i++)
    {
        for(int j = 0; j < HSplit; j++)
        {
            Rect rect00(i * cropSize, j * cropSize, cropSize, cropSize);   //四个参数代表x,y,width,height
            Mat img_cut = Mat(img, rect00);
            Mat imgCrop = img_cut.clone();
            Mat imgCrop_flatten = imgCrop.reshape(0, 1);
            if(i < WSplit / 2)
            {
                for (int k = 0; k < imgCrop_flatten.cols; k++)
                {
                    TrainDataF[j * (WSplit / 2) + i][k] = (float)imgCrop_flatten.data[k];
                }
                TrainLabelF[j * (WSplit / 2) + i] = (int)j/5;
            }
            else
            {
                for (int k = 0; k < imgCrop_flatten.cols; k++)
                {
                    ValDataF[j * (WSplit / 2) + i - (WSplit / 2)][k] = (float)imgCrop_flatten.data[k];
                }
                ValLabelF[j * (WSplit / 2) + i - (WSplit / 2)] = (int)j/5;
            }
        }
    }
    TrainData = Mat(trainTotal, cropSize * cropSize, CV_32FC1, TrainDataF);
    TrainLabel = Mat(trainTotal, 1, CV_32FC1, TrainLabelF);
    ValData = Mat(trainTotal, cropSize * cropSize, CV_32FC1, ValDataF);
    ValLabel = Mat(trainTotal, 1, CV_32FC1, ValLabelF);
}

void TrainDigit::train()
{
    kclassifier = KNearest::create();
    kclassifier->setDefaultK(5);
    kclassifier->setIsClassifier(true);
    Ptr<ml::TrainData>trainData = TrainData::create(TrainData, ROW_SAMPLE, TrainLabel);
    std::cout << "[start] train model" << std::endl;
    kclassifier->train(trainData);
}

void TrainDigit::validation()
{
}

void TrainDigit::save_model(string out_model_path)
{
    kclassifier->save(out_model_path);
}
