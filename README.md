# openvino-rm-yolo-face
openvino inference based on yolov5-face for rm

# requirements
c++:
- openvino-runtime 2022
- opencv

python:
- openvino
- torch
- torchvision

# usage
**python:**   
In your python env
```python
python main.py
```
**c++:**
```shell
cd cpp && mkdir build
cd build
cmake ..
make 
```