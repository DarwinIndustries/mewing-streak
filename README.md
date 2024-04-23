# mouth-open
Detecting when a human's mouth is open but when it is your mac shuts down

`sudo python3 main.py` to run

## Necessary Install
[download shape face .dat necessary](https://www.mediafire.com/file/onwfax3xk64lzmz/shape_predictor_68_face_landmarks.dat/file)
> Keep in same directory as the main.py file

## Referenced Code

[Adrian Rosebrock - drowsiness detection](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)

I was interested in implementing a similar function for calculating the aspect ratio of the mouth instead of both eyes. 

## Dependencies
Python modules for:
* scipy
* imutils
* numpy
* dlib
* cv2

## Usage
This sample version uses your webcam, so make sure that the device you are using has one.  Otherwise, you will need to change the code to take in a video file.

[dlib shape predictor](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat)
