## SSD Score evaluation.

1. Use docker to evaluate.
``` 
docker pull linkernetworks/caffe-ssd:1.0
```
or with GPUs
``` 
docker pull linkernetworks/caffe-ssd:1.0-gpu
```

2. And download weights from Google Drive.

https://drive.google.com/file/d/0BzKzrI_SkD1_WVVTSmQxU0dVRzA/view?resourcekey=0-VFoYwzlHsh486pZ5U4sA6w
Copy VGG_VOC0712_SSD_300x300_iter_120000.caffemodel to this folder.

3. uncomment the code with caffe.set_device(0) and caffe.set_mode_gpu() if you are using cpu version.

4. use the script by cmd:
```
python3 compute_ssd_score_XXXX.py --test_dir your/test/dir 
```

If you are using CPU version docker, the test time would be 3 items/second

Note: be careful, your result should be of our output format, or the evaluation result would not correspond with ours.
