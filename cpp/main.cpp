#include <iostream>
#include <chrono>
#include "ovinference.h"

int main(void) {

    cv::VideoCapture video("/home/quonone/Videos/video.mp4");

    std::shared_ptr<OvInference> ovinfer = std::make_shared<OvInference>("../../model/rm-net16.xml");

    while (1) {
        cv::Mat src;
        video.read(src);
        if (src.empty())
            break;
        auto t0 = std::chrono::steady_clock::now();
        ovinfer->infer(src);
        auto t1 = std::chrono::steady_clock::now();
        float per = std::chrono::duration<double, std::milli>(t1 - t0).count();
        cv::putText(src, std::to_string(per) + " ms", cv::Point(15, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                    cv::Scalar(255, 0, 0));
        cv::imshow("result", src);
        cv::waitKey(1);
    }

}