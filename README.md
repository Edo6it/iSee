# iSee
Repo for CV exam project.
iSee is a custom yolov4 network whose aim is to help visually impaired people to cross the street and monitoring the surrounding urban environment. 

## Getting Started
### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

## Custom YOLOv4 Using TensorFlow
The following commands will allow you to run your custom yolov4 model. (video and webcam commands work as well)
```
# custom yolov4
python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4 

# Run custom yolov4 on image
python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images PATH_TO_IMAGE

# Run yolov4 on video
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video PATH_TO_VIDEO --output ./detections/results.mp4
```

### Result Image(s)
You can find the outputted image(s) showing the detections saved within the 'detections' folder.

### Result Video
Video saves wherever you point --output flag to. If you don't set the flag then your video will not be saved with detections on it.

### Counting Objects (total objects or per class)
We have created a custom function within the file [core/functions.py](https://github.com/theAIGuysCode/yolov4-custom-functions/blob/master/core/functions.py) that can be used to count and keep track of the number of people detected at a given moment within each image or video. It can be used to monitor the crowding during the street crossing. 

To use this function all that is needed is to add the custom flag "--count" to your detect.py or detect_video.py command.

### References  

   Huge shoutout goes to hunglc007 for creating the backbone of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
