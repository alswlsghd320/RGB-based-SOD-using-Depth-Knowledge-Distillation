# Approach to improve RGB-based Salient Object Detection using Depth Information and Knowledge Distillation
2021년 한국지능정보시스템학회 추계 학술대회 우수 논문상 ([website](http://www.kiiss.or.kr/conference/conf/sub05.html))

## Author
* [JinHong Min](https://github.com/alswlsghd320)
* [Eunbi Park](https://github.com/bluvory)
* Hongju Lee

Paper: *<upload later>*<br>

  <i><b>Abstract</b>: Unlike conventional object detection, Salient Object Detection aims to find the most important object in the image. That is, by separating an object in which human gaze is concentrated from the background, it is used in various fields such as background removal, object tracking and detection, and recognition. The detection of protruding objects is mainly performed in RGB data to selectively grasp global and regional situation information. However, in the case of less contrast, detection may fail, so it plays an important role in finding prominent objects by using additional depth information to compensate for this. However, as opposed to improved performance, additional depth information increases the complexity of the model and lowers the inference speed. Therefore, there is a need to use depth information, but solve the above problems. This study proposes a method to improve the accuracy in detecting protruding objects using Knowledge Distillation. We first learn the RGBD model and then use it as a teacher network. After that, the RGB model is placed as a student neural network and knowledge distillation is applied to increase the performance of the existing RGB model. In addition, the hyperparameters used in knowledge distillation are not set to fixed values, but stable performance improvement is achieved through the method proposed in this paper. Through this, it is possible to overcome problems in practical aspects such as limitations of memory and increase in inference time as well as improvement of detection performance. In this experiment, RGB-based protruding object detection models such as BASNet, U2net, and PoolNet were evaluated through a total of four evaluation indicators: MAE, F-measure, E-measure, and S-measure, showing meaningful performance improvement of models using knowledge distillation over conventional RGB models.</i>
  
<i><b>Keywords</b>: Salient Object Detection, Knowledge Distillation, Removing Background</i>

## Requirements
```.bash
#For Anaconda
conda env create -f environment_cuda11_1.yml # For CUDA 11.1
```

## Training
We train our models with
  [NJU2K](https://paperswithcode.com/dataset/nju2k),
  [SIP](https://paperswithcode.com/dataset/sip),
  [DUT-RGBD/train]()

```.bash
'''
KIISS2021/
  ㄴdatasets/
        ㄴDUT-RGBD/
        ㄴLFSD/
        ㄴNJU2K/
        ㄴSIP/
        ...
  ㄴmodels/
  ...
'''
# For Train Teacher Network
python train_rgbd.py --args

# For Train Student Network Only
python train_rgb.py --args

# For Train Student Network with Knowledge Distillation
python train_kd.py --args
```

The results are placed in `run/<RUNNING_TIME>/<tensorboard>`. 

## Test

We train our models with
  [SSD](),
  [LFSD](https://paperswithcode.com/dataset/lfsd),
  [STEREO-1000](),
  [DUT-RGBD/test]()

```.bash
# For test
python test.py --path <model_path> --model <PoolNet or BASNet or u2net> --dataset <test_dataset> ...
```

## Results
### Quantitative Comparison (Knowledge Distillation)
![table1](https://github.com/alswlsghd320/KIISS2021/blob/main/figure/table1.png)
### Quantitative Comparison (hypterparameter &alpha;)
![table2](https://github.com/alswlsghd320/KIISS2021/blob/main/figure/table2.png)
### Model Architecture
![figure1](https://github.com/alswlsghd320/KIISS2021/blob/main/figure/figure1.png)
### Qualitative Comparison (Knowledge Distillation)
![figure2](https://github.com/alswlsghd320/KIISS2021/blob/main/figure/figure2.png)
### Qualitative Comparison (Depth Knowledge Distillation)
![figure3](https://github.com/alswlsghd320/KIISS2021/blob/main/figure/figure3.png)

## Citation
  <pre><code></code></pre>

## Contact us
If you have any questions, please contact us (alswlsghd320@naver.com)
