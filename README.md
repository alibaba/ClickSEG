# ClickSEG: A Codebase for Click-Based Interactive Segmentation
## Introduction 
ClickSEG is codebase for click-based interactive segmentation developped on [RITM codebase][ritmcode]. 

## What's New?
Compared with the repo of [RITM codebase][ritmcode], ClickSEG has following new features:

#### **1. The official implementation for the following papers.** 
> Conditional Diffusion for Interative Segmentation (ICCV2021) \[[Link][cdnet]\]  
> FocalClick: Towards Practical Interactive Image Segmentation (CVPR2022) 


#### **2. More correct crop augmentation during training.**
[RITM codebase][ritmcode] uses [albumentations][albumentations] to crop and resize image-mask pairs for training. In this way, the crop size are fixed, which is not suitable for training on a combined dataset with variant image size; Besides, the NEAREST INTERPOLATION adopt in [albumentations][albumentations] causes the mask to have 1 pixel bias towards bottom-right, which is harmful for the boundary details, especially for the Refiner of FocalClick. 

Therefore, we re-write the augmentation, which is crucial for the final performance. 



#### **3. More backbones and more train/val data.**
We add efficient backbones like [MobileNets][1] and [PPLCNet][2]. We trained all our models on COCO+LVIS dataset for the standard configuration. At the same time, we train them on a combinatory large dataset and provide the trained weight to facilitate academic research and industrial applications. The combinatory large dataset include 8 dataset with high quality annotations and Diversified scenes:  COCO[<sup>1</sup>](#coco), LVIS[<sup>2</sup>](#lvis), ADE20K[<sup>3</sup>](#ade20k), MSRA10K[<sup>4</sup>](#msra10k), DUT[<sup>5</sup>](#dut), YoutubeVOS[<sup>6</sup>](#ytbvos), ThinObject[<sup>7</sup>](#thin), HFlicker[<sup>8</sup>](#Hflicker).

```
1. Microsoft coco: Common objects in context
2. Lvis: A dataset for large vocabulary instance segmentation
3. Scene Parsing through ADE20K Dataset
4. Salient object detection: A benchmark
5. Learning to detect salient objects with image-level supervision
6. YouTube-VOS: A Large-Scale Video Object Segmentation Benchmark
7. Deep Interactive Thin Object Selection
8. DoveNet: Deep Image Harmonization via Domain Verification
```

#### **4. Dataset and evaluation code for starting from initial masks.**
In the paper of FocalClick, we propose a new dataset of DAVIS-585 which provides initial masks for evaluation. The dataset could be download at [ClickSEG GOOGLE DIRVIE][drive]. We also provide evaluation code in this codebase.


[ritmcode]:https://github.com/saic-vul/ritm_interactive_segmentation
[albumentations]:https://albumentations.ai/
[cdnet]: https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Conditional_Diffusion_for_Interactive_Segmentation_ICCV_2021_paper.pdf 

[focalclick]: https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Conditional_Diffusion_for_Interactive_Segmentation_ICCV_2021_paper.pdf 

<br/>
<br/>

## User Guidelines
To use this codebase to train/val your own models, please follow the steps:

1. Install the requirements by excuting
```
pip install -r requirements.txt
```

2. Prepare the dataset and pretrained backbone weights following: [Data_Weight_Preparation.md](assets/documents/Data_Weight_Preparation.md)

3. Train or validate the model following: [Train_Val_Guidance.md](assets/documents/Train_Val_Guidance.md)


<br/>
<br/>

## Supported Methods

The trained model weights could be downloaded at [ClickSEG GOOGLE DIRVIE][drive]


### CDNet: Conditional Diffusion for Interative Segmentation (ICCV2021)
```
CONFIG
Input Size: 384 x 384
Previous Mask: No
Iterative Training: No
```
<table>
    <thead align="center">
        <tr>
            <th rowspan="2"><span style="font-weight:bold">Train</span><br><span style="font-weight:bold">Dataset</span></th>
            <th rowspan="2">Model</th>
            <th>GrabCut</th>
            <th>Berkeley</th>
            <th>Pascal<br>VOC</th>
            <th>COCO<br>MVal</th>
            <th>SBD</th>    
            <th>DAVIS</th>
            <th>DAVIS585<br>from zero</th>
            <th>DAVIS585<br>from init</th>
        </tr>
        <tr>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td rowspan="1">SBD</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">ResNet34<br>(89.72 MB)</a></td>
            <td>1.86/2.18</td>
            <td>1.95/3.27</td>
            <td>3.61/4.51</td>
            <td>4.13/5.88</td>
            <td>5.18/7.89</td>
            <td>5.00/6.89</td>
            <td>6.68/9.59</td>
            <td>5.04/7.06</td>
        </tr>
    </tbody>
    <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">ResNet34<br>(89.72 MB)</a></td>
            <td>1.40/1.52</td>
            <td>1.47/2.06</td>
            <td>2.74/3.30</td>
            <td>2.51/3.88</td>
            <td>4.30/7.04</td>
            <td>4.27/5.56</td>
            <td>4.86/7.37</td>
            <td>4.21/5.92</td>
        </tr>
    </tbody>
</table>




### FocalClick: Towards Practical Interactive Image Segmentation (CVPR2022)
```
CONFIG
S1 version: coarse segmentator input size 128x128; refiner input size 256x256.  
S2 version: coarse segmentator input size 256x256; refiner input size 256x256.  
Previous Mask: Yes
Iterative Training: Yes
```


<table>
    <thead align="center">
        <tr>
            <th rowspan="2"><span style="font-weight:bold">Train</span><br><span style="font-weight:bold">Dataset</span></th>
            <th rowspan="2">Model</th>
            <th>GrabCut</th>
            <th>Berkeley</th>
            <th>Pascal<br>VOC</th>
            <th>COCO<br>MVal</th>
            <th>SBD</th>    
            <th>DAVIS</th>
            <th>DAVIS585<br>from zero</th>
            <th>DAVIS585<br>from init</th>
        </tr>
        <tr>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
        </tr>
    </thead>
        <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">HRNet18s-S1<br>(16.58 MB)</a></td>
            <td>1.64/1.88</td>
            <td>1.84/2.89</td>
            <td>3.24/3.91</td>
            <td>2.89/4.00</td>
            <td>4.74/7.29</td>
            <td>4.77/6.56</td>
            <td>5.62/8.08</td>
            <td>2.72/3.82</td>
        </tr>
    </tbody>
     <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">HRNet18s-S2<br>(16.58 MB)</a></td>
            <td>1.48/1.62</td>
            <td>1.60/2.23</td>
            <td>2.93/3.46</td>
            <td>2.61/3.59</td>
            <td>4.43/6.79</td>
            <td>3.90/5.23</td>
            <td>4.87/6.87</td>
            <td>2.47/3.30</td>
        </tr>
    </tbody>
    <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">HRNet32-S2<br>(119.11 MB)</a></td>
            <td>1.64/1.80</td>
            <td>1.70/2.36</td>
            <td>2.80/3.35</td>
            <td>2.62/3.65</td>
            <td>4.24/6.61</td>
            <td>4.01/5.39</td>
            <td>4.77/6.84</td>
            <td>2.32/3.09</td>
        </tr>
    </tbody>
         <tbody align="center">
        <tr>
            <td rowspan="1">Combined+<br>Dataset</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">HRNet32-S2<br>(119.11 MB)</a></td>
            <td>1.30/1.34</td>
            <td>1.49/1.85</td>
            <td>2.84/3.38</td>
            <td>2.80/3.85</td>
            <td>4.35/6.61</td>
            <td>3.19/4.81</td>
            <td>4.80/6.63</td>
            <td>2.37/3.26</td>
        </tr>
    </tbody>
    <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">SegFormerB0-S1<br>(14.38 MB)</a></td>
            <td>1.60/1.86</td>
            <td>2.05/3.29</td>
            <td>3.54/4.22</td>
            <td>3.08/4.21</td>
            <td>4.98/7.60</td>
            <td>5.13/7.42</td>
            <td>6.21/9.06</td>
            <td>2.63/3.69</td>
        </tr>
    </tbody>
    <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">SegFormerB0-S2<br>(14.38 MB)</a></td>
            <td>1.40/1.66</td>
            <td>1.59/2.27</td>
            <td>2.97/3.52</td>
            <td>2.65/3.59</td>
            <td>4.56/6.86</td>
            <td>4.04/5.49</td>
            <td>5.01/7.22</td>
            <td>2.21/3.08</td>
        </tr>
    </tbody>
    <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">SegFormerB3-S2<br>(174.56 MB)</a></td>
            <td>1.44/1.50</td>
            <td>1.55/1.92</td>
            <td>2.46/2.88</td>
            <td>2.32/3.12</td>
            <td>3.53/5.59</td>
            <td>3.61/4.90</td>
            <td>4.06/5.89</td>
            <td>2.00/2.76</td>
        </tr>
    </tbody>
    <tbody align="center">
        <tr>
            <td rowspan="1">Combined<br>Datasets</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">SegFormerB3-S2<br>(174.56 MB)</a></td>
            <td>1.22/1.26</td>
            <td>1.35/1.48</td>
            <td>2.54/2.96</td>
            <td>2.51/3.33</td>
            <td>3.70/5.84</td>
            <td>2.92/4.52</td>
            <td>3.98/5.75</td>
            <td>1.98/2.72</td>
        </tr>
    </tbody>
</table>




### Efficient Baselines using [MobileNets][1] and [PPLCNets][2]
```
CONFIG
Input Size: 384x384.
Previous Mask: Yes
Iterative Training: Yes
```
<table>
    <thead align="center">
        <tr>
            <th rowspan="2"><span style="font-weight:bold">Train</span><br><span style="font-weight:bold">Dataset</span></th>
            <th rowspan="2">Model</th>
            <th>GrabCut</th>
            <th>Berkeley</th>
            <th>Pascal<br>VOC</th>
            <th>COCO<br>MVal</th>
            <th>SBD</th>    
            <th>DAVIS</th>
            <th>DAVIS585<br>from zero</th>
            <th>DAVIS585<br>from init</th>
        </tr>
        <tr>
           <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">MobileNetV2<br>(7.5 MB)</a></td>
            <td>1.82/2.02</td>
            <td>1.95/2.69</td>
            <td>2.97/3.61</td>
            <td>2.74/3.73</td>
            <td>4.44/6.75</td>
            <td>3.65/5.81</td>
            <td>5.25/7.28</td>
            <td>2.15/3.04</td>
        </tr>
    </tbody>
        <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">PPLCNet<br>(11.92 MB)</a></td>
            <td>1.74/1.92</td>
            <td>1.96/2.66</td>
            <td>2.95/3.51</td>
            <td>2.72/3.75</td>
            <td>4.41/6.66</td>
            <td>4.40/5.78</td>
            <td>5.11/7.28</td>
            <td>2.03/2.90</td>
        </tr>
    </tbody>
    <tbody align="center">
        <tr>
            <td rowspan="1">Combined<br>Datasets</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">MobileNetV2<br>(7.5 MB)</a></td>
            <td>1.50/1.62</td>
            <td>1.62/2.25</td>
            <td>3.00/3.61</td>
            <td>2.80/3.96</td>
            <td>4.66/7.05</td>
            <td>3.59/5.24</td>
            <td>5.05/7.12</td>
            <td>2.06/2.97</td>
        </tr>
    </tbody>
        <tbody align="center">
        <tr>
            <td rowspan="1">Combined<br>Datasets</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">PPLCNet<br>(11.92 MB)</a></td>
            <td>1.46/1.66</td>
            <td>1.63/1.99</td>
            <td>2.88/3.44</td>
            <td>2.75/3.89</td>
            <td>4.44/6.74</td>
            <td>3.65/5.34</td>
            <td>5.02/6.98</td>
            <td>1.96/2.81</td>
        </tr>
    </tbody>
</table>







<br/>
<br/>

## License

The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source. 

<br/>
<br/>

## Acknowledgement
The core framework of this codebase follows: https://github.com/saic-vul/ritm_interactive_segmentation

Some code and pretrained weights are brought from:  
 https://github.com/Tramac/Lightweight-Segmentation  
 https://github.com/facebookresearch/video-nonlocal-net  
 https://github.com/visinf/1-stage-wseg  
 https://github.com/frotms/PP-LCNet-Pytorch  

We thank those authors for their great works.

<br/>
<br/>

## Citation

If you find this work is useful for your research, please cite our papers:
```
@inproceedings{cdnet,
  title={Conditional Diffusion for Interactive Segmentation},
  author={Chen, Xi and Zhao, Zhiyan and Yu, Feiwu and Zhang, Yilei and Duan, Manni},
  booktitle={ICCV},
  year={2021}
}

@article{focalclick,
  title={FocalClick: Towards Practical Interactive Image Segmentation},
  author={Chen, Xi and Zhao, Zhiyan and Zhang, Yilei and Duan, Manni and Qi, Donglian and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2022}
}
```

[1]:https://arxiv.org/abs/1801.04381
[2]:https://arxiv.org/abs/2109.15099
[drive]:https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing
