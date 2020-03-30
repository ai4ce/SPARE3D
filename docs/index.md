# SPARE3D: A Dataset for SPAtial REasoning on Three-View Line Drawings

[**Wenyu Han** (NYU Tandon School of Engineering)](https://ai4ce.github.io), [**Siyuan Xiang** (NYU Tandon School of Engineering)](https://ai4ce.github.io),[**Chen Feng** (NYU Tandon School of Engineering)](https://ai4ce.github.io),[**Chenhui Liu** (NYU Tandon School of Engineering)](https://ai4ce.github.io),[**Ruoyu Wang** (NYU Tandon School of Engineering)](https://ai4ce.github.io)

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2020

|[Abstract](#abstract)|[Paper](#paper-arxiv)|[Code](#code-github)|[Dataset](#dataset)|[Results](#results)|[Acknowledgment](#acknowledgment)|
#### Task view point settings 
![Pose](https://github.com/ai4ce/SPARE3D/blob/master/docs/figs/view_point.jpg)
#### Three View to Isometric task 
![Three View to Iso task](https://github.com/ai4ce/SPARE3D/blob/master/docs/figs/Three_view_to_iso.jpg)
#### Isometric to Pose task 
![Iso to pose task](https://github.com/ai4ce/SPARE3D/blob/master/docs/figs/Iso_to_pose.png)
#### Pose to Isometric task 
![Pose to iso task](https://github.com/ai4ce/SPARE3D/blob/master/docs/figs/Pose_to_iso.jpg)

### Abstract
Spatial reasoning is an important component of human intelligence. We can imagine the shapes of 3D objects and reason about their spatial relations by merely looking at their three-view line drawings in 2D, with different levels of competence. Can deep networks be trained to perform spatial reasoning tasks? How can we measure their ``spatial intelligence''? To answer these questions, we present the SPARE3D dataset. Based on cognitive science and psychometrics, SPARE3D contains three types of 2D-3D reasoning tasks on view consistency, camera pose, and shape generation, with increasing difficulty. We then design a method to automatically generate a large number of challenging questions with ground truth answers for each task. They are used to provide supervision for training our baseline models using state-of-the-art architectures like ResNet. Our experiments show that although convolutional networks have achieved superhuman performance in many visual learning tasks, their spatial reasoning performances in SPARE3D are almost equal to random guesses. We hope SPARE3D can stimulate new problem formulations and network designs for spatial reasoning to empower intelligent robots to operate effectively in the 3D world via 2D sensors

### [Paper (arXiv)]()
To cite our paper:

```
```

### [Code (GitHub)](https://github.com/ai4ce/spare3d)

```

``` 
### Dataset

Please download dataset via [google drive link](https://drive.google.com/open?id=1Mi2KZyKAlUOGYRQTDz3E5nhiXY5GhUB2). This google drive folder contains three zip files: Task_data.zip is for training baseline; CSG_model_step.zip contains 11149 CSG models; Total_view_data contains view drawings of all ABC and CSG models from 11 pose we define in the paper.   
If you meet any problems, please contact [authors](https://ai4ce.github.io) for help. 

### Results
#### SPARE3D benchmark results of Three View to Isometric, Isometric to Pose, and Pose to Isometric tasks
![Baseline_barchart](https://github.com/ai4ce/SPARE3D/blob/master/docs/figs/baseline_barchart.PNG)
#### Isometric View Generation task testing samples
![Isometric view generation result](https://github.com/ai4ce/SPARE3D/blob/master/docs/figs/Iso_view_generation.PNG)
#### Point Cloud Generation task testing samples
![Point cloud generation result](https://github.com/ai4ce/SPARE3D/blob/master/docs/figs/point_cloud_generation.PNG)


### Acknowledgment

This project is done equally by [Wenyu Han](https://ai4ce.github.io) and [Siyuan Xiang](https://ai4ce.github.io). Thank for the contributions made by Chenhui liu (Building website for collecting human performance data) and Ruoyu Wang(Code and paper modification) 

