cmake_minimum_required(VERSION 3.24)
project(ares)

include_directories(
        #OpenVINO推理引擎的头文件
        /opt/intel/openvino2022/runtime/include/
)

#查找必要的依赖包
find_package(OpenCV REQUIRED)
set(InferenceEngine_DIR "/opt/intel/openvino2022/runtime/cmake")
find_package(InferenceEngine)
set(ngraph_DIR "/opt/intel/openvino2022/runtime/cmake")
find_package(ngraph REQUIRED)

# 编译detector.so



add_executable(detect main.cpp)
target_link_libraries(detect
        ${OpenCV_LIBS}
        openvino::runtime
        )