# SPARE3D: A Dataset for SPAtial REasoning on Three-View Line Drawings

[**Wenyu Han**](https://github.com/WenyuHan-LiNa), [**Siyuan Xiang**](https://www.linkedin.com/in/%E6%80%9D%E8%BF%9C-%E9%A1%B9-b4b920145/), [**Chenhui Liu**](https://github.com/iamshenkui), [**Ruoyu Wang**](https://github.com/ruoyuwangeel4930), [**Chen Feng**](https://engineering.nyu.edu/faculty/chen-feng)

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2020

[New York University Tandon School of Engineering](https://ai4ce.github.io)

![Overview](https://github.com/ai4ce/SPARE3D/raw/master/docs/figs/task_overview.jpg)

|[Abstract](#abstract)|[Dataset](#dataset)|[Code](#code-github)|[Paper](#paper-arxiv)|[Results](#results)|[Acknowledgment](#acknowledgment)|

### Abstract
Spatial reasoning is an important component of human intelligence. We can imagine the shapes of 3D objects and reason about their spatial relations by merely looking at their three-view line drawings in 2D, with different levels of competence. Can deep networks be trained to perform spatial reasoning tasks? How can we measure their "spatial intelligence"? To answer these questions, we present the SPARE3D dataset. Based on cognitive science and psychometrics, SPARE3D contains three types of 2D-3D reasoning tasks on view consistency, camera pose, and shape generation, with increasing difficulty. We then design a method to automatically generate a large number of challenging questions with ground truth answers for each task. They are used to provide supervision for training our baseline models using state-of-the-art architectures like ResNet. Our experiments show that although convolutional networks have achieved superhuman performance in many visual learning tasks, their spatial reasoning performance in SPARE3D is almost equal to random guesses. We hope SPARE3D can stimulate new problem formulations and network designs for spatial reasoning to empower intelligent robots to operate effectively in the 3D world via 2D sensors.

### Dataset
You can download the dataset via [our google drive link](https://drive.google.com/drive/folders/1Mi2KZyKAlUOGYRQTDz3E5nhiXY5GhUB2?usp=sharing). This google drive folder contains three zip files: 
1. Task_data.zip is for training baseline;
2. CSG_model_step.zip contains 11149 CSG models;
3. Total_view_data contains view drawings of all ABC and CSG models from 11 pose we define in the paper.

Please feel free to report bugs or other problems to [the authors](https://ai4ce.github.io).

### [Code (GitHub)](https://github.com/ai4ce/spare3d)
```
The code is copyrighted by the authors. Permission to copy and use 
 this software for noncommercial use is hereby granted provided: (a)
 this notice is retained in all copies, (2) the publication describing
 the method (indicated below) is clearly cited, and (3) the
 distribution from which the code was obtained is clearly cited. For
 all other uses, please contact the authors.
 
 The software code is provided "as is" with ABSOLUTELY NO WARRANTY
 expressed or implied. Use at your own risk.

This code provides an implementation of the method described in the
following publication: 

Wenyu Han, Siyuan Xiang, Chenhui Liu, Ruoyu Wang, and Chen Feng, 
"SPARE3D: A Dataset for SPAtial REasoning on Three-View Line Drawings," 
The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June, 2020.
``` 

### [Paper (arXiv)](https://github.com/ai4ce/SPARE3D/raw/master/docs/SPARE3D.pdf)
To cite our paper:

```
@InProceedings{SPARE3D_CVPR_2020,
author = {Han, Wenyu and Xiang, Siyuan and Liu, Chenhui and Wang, Ruoyu and Feng, Chen},
title = { {SPARE3D}: A Dataset for {SPA}tial {RE}asoning on Three-View Line Drawings},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

#### Task viewpoint settings 
![Pose](https://github.com/ai4ce/SPARE3D/raw/master/docs/figs/view_point.jpg)
#### Three View to Isometric task 
![Three View to Iso task](https://github.com/ai4ce/SPARE3D/raw/master/docs/figs/Three_view_to_iso.jpg)
#### Isometric to Pose task 
![Iso to pose task](https://github.com/ai4ce/SPARE3D/raw/master/docs/figs/Iso_to_pose.jpg)
#### Pose to Isometric task 
![Pose to iso task](https://github.com/ai4ce/SPARE3D/raw/master/docs/figs/Pose_to_iso.jpg)




### Results
#### SPARE3D benchmark results of Three View to Isometric, Isometric to Pose, and Pose to Isometric tasks
**Top row is for SPARE3D-ABC, bottom row is for SPARE3D-CSG.**
![Baseline_barchart](https://github.com/ai4ce/SPARE3D/raw/master/docs/figs/baseline_barchart.PNG)

#### Isometric View Generation task testing samples
![Isometric view generation result](https://github.com/ai4ce/SPARE3D/raw/master/docs/figs/Iso_view_generation.PNG)

#### Point Cloud Generation task testing samples
![Point cloud generation result](https://github.com/ai4ce/SPARE3D/raw/master/docs/figs/point_cloud_generation.PNG)


### Acknowledgment
[**Wenyu Han**](https://github.com/WenyuHan-LiNa) and [**Siyuan Xiang**](https://www.linkedin.com/in/%E6%80%9D%E8%BF%9C-%E9%A1%B9-b4b920145/) contributed equally to the coding, data preprocessing/generation, paper writing, and experiments in this project. [**Chenhui Liu**](https://github.com/iamshenkui) contributed to the crowd-sourcing website and human performance data collection. [**Ruoyu Wang**](https://github.com/ruoyuwangeel4930) contributed to the experiments and paper writing. [**Chen Feng**](https://ai4ce.github.io) proposed the idea, initiated the project, and contributed to the coding and paper writing.

The research is supported by [NSF CPS program under CMMI-1932187](https://nsf.gov/awardsearch/showAward?AWD_ID=1932187). Siyuan Xiang gratefully thanks the IDC Foundation for its scholarship. The authors gratefully thank our human test participants and the helpful comments from Zhaorong Wang, Zhiding Yu, Srikumar Ramalingam, and the anonymous reviewers.
