//
// Created by quanyi on 23-2-16.
//
#include <opencv2/dnn.hpp>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;

///@brief: openvino 测试，待优化的地方：设置结果遍历最大限制参数，减少耗时
///TODO: 封装成类直接调用；部分代码写得繁琐，适当简化；FP16的使用，现在是FP32

const float SCORE_THRESHOLD = 0.6;
const float NMS_THRESHOLD = 0.5;
const float CONFIDENCE_THRESHOLD = 0.6;

struct Armor{
    cv::Rect box;
    cv::Point2f p1;
    cv::Point2f p2;
    cv::Point2f p3;
    cv::Point2f p4;
};

struct Detection {
    int class_id;
    float confidence;
    Armor armor;
};


struct Resize {
    cv::Mat resized_image;
    int dw;
    int dh;
};


Resize resize_and_pad(cv::Mat &img, cv::Size new_shape) {
    float width = img.cols;
    float height = img.rows;
    float r = float(new_shape.width / max(width, height));
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    Resize resize;
    cv::resize(img, resize.resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);

    resize.dw = new_shape.width - new_unpadW;
    resize.dh = new_shape.height - new_unpadH;
    cv::Scalar color = cv::Scalar(100, 100, 100);
    cv::copyMakeBorder(resize.resized_image, resize.resized_image, 0, resize.dh, 0, resize.dw, cv::BORDER_CONSTANT,
                       color);

    return resize;
}


int main() {

    // Step 1. Initialize OpenVINO Runtime core
    ov::Core core;
    // Step 2. Read a model
    std::shared_ptr<ov::Model> model = core.read_model("../../model/rm-net16.xml");


    // Step 3. Read input image
    cv::Mat img = cv::imread("../../img/aaa.jpg");
    // resize image
    Resize res = resize_and_pad(img, cv::Size(640, 640));


    // Step 4. Inizialize Preprocessing for the model
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    // Specify input image format
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(
            ov::preprocess::ColorFormat::BGR);
    // Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(
            ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});
    //  Specify model's input layout
    ppp.input().model().set_layout("NCHW");
    // Specify output results format
    ppp.output().tensor().set_element_type(ov::element::f32);
    // Embed above steps in the graph
    model = ppp.build();
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");

    int i = 0;
    //循环10次测试帧率
    while(i<10){
        i++;
        auto t0 = std::chrono::steady_clock::now();
        // Step 5. Create tensor from image
        float *input_data = (float *) res.resized_image.data;
        ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(),
                                             input_data);


        // Step 6. Create an infer request for model inference
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();


        //Step 7. Retrieve inference results
        const ov::Tensor &output_tensor = infer_request.get_output_tensor();
        ov::Shape output_shape = output_tensor.get_shape();
        float *detections = output_tensor.data<float>();

        std::cout<< " detections "<< output_tensor.get_shape()<<std::endl;
        auto t1 = std::chrono::steady_clock::now();
        // Step 8. Postprocessing including NMS
        std::vector<cv::Rect> boxes; //用于NMS
        std::vector<Armor> armors;
        vector<int> class_ids;
        vector<float> confidences;
        //遍历每一个候选结果
        // x,y,w,h,conf, x1,y1,x2,y2,x3,y3,x4,y4, c1,c2,c3, ... , c36
        for (int i = 0; i < output_shape[1]; i++) {
            float *detection = &detections[i * output_shape[2]];
            float confidence = detection[4];
            //选择置信度大于0.6的
            if (confidence >= CONFIDENCE_THRESHOLD) {
                //对于原来的yolo5,前五个后面是各个的类别
                // 但是对于yoloface，应该是第13个开始
                float *classes_scores = &detection[13];
                //生成每一个类别对应分数的矩阵，
                cv::Mat scores(1, output_shape[2] - 13, CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;
                //找到最大分数的位置，即预测类别
                cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                //如果大于分数
                if (max_class_score > SCORE_THRESHOLD) {
                    //按原来的顺序，存储下来（置信度）
                    confidences.push_back(confidence);
                    //每一个对应的类别标签的位置
                    class_ids.push_back(class_id.x);

                    float x = detection[0];
                    float y = detection[1];
                    float w = detection[2];
                    float h = detection[3];
                    cv::Point2f p1 = cv::Point2f (detection[5],detection[6]);
                    cv::Point2f p2 = cv::Point2f (detection[7],detection[8]);
                    cv::Point2f p3 = cv::Point2f (detection[9],detection[10]);
                    cv::Point2f p4 = cv::Point2f (detection[11],detection[12]);

                    float xmin = x - (w / 2);
                    float ymin = y - (h / 2);

                    boxes.push_back(cv::Rect(xmin, ymin, w, h));
                    Armor armor = {cv::Rect(xmin,ymin,w,h), p1, p2, p3, p4};
                    armors.push_back(armor);
                }
            }
        }

        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

        std::vector<Detection> output;
        for (int i = 0; i < nms_result.size(); i++) {
            Detection result;
            int idx = nms_result[i];
            result.class_id = class_ids[idx];
            result.confidence = confidences[idx];
            result.armor = armors[idx];
            output.push_back(result);
        }


        // Step 9. Print results and save Figure with detections
        for (int i = 0; i < output.size(); i++) {
            auto detection = output[i];
            auto box = detection.armor.box;
            auto p1 = detection.armor.p1;
            auto p2 = detection.armor.p2;
            auto p3 = detection.armor.p3;
            auto p4 = detection.armor.p4;
            auto classId = detection.class_id;
            auto confidence = detection.confidence;
            float rx = (float) img.cols / (float) (res.resized_image.cols - res.dw);
            float ry = (float) img.rows / (float) (res.resized_image.rows - res.dh);
            box.x *= rx ;
            box.y *= ry ;
            box.width *= rx;
            box.height *= ry;
            p1.x *= rx;
            p1.y *= ry;
            p2.x *= rx;
            p2.y *= ry;
            p3.x *= rx;
            p3.y *= ry;
            p4.x *= rx;
            p4.y *= ry;


            cout << "Bbox" << i + 1 << ": Class: " << classId << " "
                 << "Confidence: " << confidence << " Scaled coords: [ "
                 << "cx: " << (float) (box.x + (box.width / 2)) / img.cols << ", "
                 << "cy: " << (float) (box.y + (box.height / 2)) / img.rows << ", "
                 << "w: " << (float) box.width / img.cols << ", "
                 << "h: " << (float) box.height / img.rows << " ]" << endl;
            float xmax = box.x + box.width;
            float ymax = box.y + box.height;
            //cv::rectangle(img, cv::Point(box.x, box.y), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 3);
            //cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(xmax, box.y), cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(img, std::to_string(classId), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0),2);
//        cv::circle(img,p1, 2, cv::Scalar (0,0,255),-1);
//        cv::circle(img,p2, 2, cv::Scalar (0,0,255),-1);
//        cv::circle(img,p3, 2, cv::Scalar (0,0,255),-1);
//        cv::circle(img,p4, 2, cv::Scalar (0,0,255),-1);

            cv::line(img, p1, p3,cv::Scalar(0,0,255),3);
            cv::line(img, p2, p4,cv::Scalar(0,0,255),3);
            cv::line(img, p1, p2,cv::Scalar(0,0,255),3);
            cv::line(img, p3, p4,cv::Scalar(0,0,255),3);
        }
        auto t2 = std::chrono::steady_clock::now();
        double per1 = std::chrono::duration<double, milli>(t1 - t0).count();
        double per2 = std::chrono::duration<double, milli>(t2 - t1).count();
        std::cout << "inference cost  " << per1 << " ms, postprocess cost " << per2 << " ms\r\n";

    }


//    cv::imwrite("./detection_cpp.png", img);
    cv::imshow("result", img);
    cv::waitKey(0);
    return 0;
}
