<p align="center">

  <h1 align="center">PanoNormal: Monocular Indoor 360° Surface Normal Estimation</h1>
  <p align="center">
    <a href="https://github.com/huangkun101230">Kun Huang</a>,
    <p>Jianwei Yang</p>,
    <p>Tielin Zhao</p>,
    <p>Lei Ji</p>,
    <p>Songyang Zhang*</p>,
    <a href="https://people.wgtn.ac.nz/fanglue.zhang?_ga=2.161972092.1710887990.1730665987-888529436.1730407824">Fang-Lue Zhang*</a>,
    <a href="https://people.wgtn.ac.nz/neil.dodgson?_ga=2.172996195.1710887990.1730665987-888529436.1730407824">Neil A. Dodgson</a>,
  </p>
    <p align="center">
    *Corresponding authors

  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2411.01749">Paper</a>
  <div align="center"></div>
</p>

## Introduction
The presence of spherical distortion in equirectangular projection (ERP) images presents a persistent challenge in dense regression tasks such as surface normal estimation. Although it may appear straightforward to repurpose architectures developed for 360° depth estimation, our empirical findings indicate that such models yield suboptimal performance when applied to surface normal prediction. This is largely attributed to their architectural bias toward capturing global scene layout, which comes at the expense of the fine-grained local geometric cues that are critical for accurate surface orientation estimation. While convolutional neural networks (CNNs) have been employed to mitigate spherical distortion, their fixed receptive fields limit their ability to capture holistic scene structure. Conversely, vision transformers (ViTs) are capable of modeling long-range dependencies via global self-attention, but often fail to preserve high-frequency local detail. To address these limitations, we propose \textit{PanoNormal}, a monocular surface normal estimation architecture for 360° images that integrates the complementary strengths of CNNs and ViTs. In particular, we design a multi-level global self-attention mechanism that explicitly accounts for the spherical feature distribution, enabling our model to recover both global contextual structure and local geometric details. Experimental results demonstrate that our method not only achieves state-of-the-art performance on several benchmark 360° datasets, but also significantly outperforms adapted depth estimation models on the task of surface normal prediction.

<p align="center">
  <a href="">
    <img src="assets/teaser.png" alt="teaser" width="95%">
  </a>
</p>
<p align="left">
Our PanoNormal method produces more accurate normal estimation predictions compared to the current state-of-the-art method, particularly in the areas highlighted by the red rectangle. For better visualization, we provide a 3D point cloud generated from the ground truth depth.</p>
<br>

<p align="center">
  <a href="">
    <img src="./assets/pipeline.png" alt="pipeline" width="95%">
  </a>
</p>
<p align="left">
Top: the overall architecture of the proposed PanoNormal method. 
Bottom: the key components: (a) The distortion-aware sampling process on the tangent patch, its transformation to the target ERP domain, and the application of a self-attention scheme among the tokens within each patch. A learnable token flow facilitates attention among the patches. (b) The proposed hierarchical multi-level transformer decoder, which produces results in different scales for comprehensive learning.</p>
<br>

## Installation
Provide installation instructions for your project. Include any dependencies and commands needed to set up the project.

```shell
# Clone the repository
git clone https://github.com/huangkun101230/PanoNormal.git
cd PanoNormal

# Install dependencies
conda env create -f conda_env.yml
conda activate panonormal
```


## Running
Please [download our pretrained models](https://drive.google.com/drive/folders/1B_GI-3mc8hgLWi0OXq-msBbGxeyuyNeB?usp=sharing), and save these models to "saved_models/models".
To test on provided data in "./input_data"
```shell
python evaluate.py
```
The results will be saved at "./results/saved_models/"

For training our model, please modify the path in our dataset:
For example, in datasets/dataset3D60.py, function gather_filepaths, change local="./input_data/" with your downloaded path

and run
```shell
python train.py
```

## Dataset
We mainly evaluate our method on [3D60 dataset](https://vcl3d.github.io/3D60/) and [Structured3D dataset](https://structured3d-dataset.org/).


## Citation
If you find this repository useful in your project, please cite the following work. :)
```
@article{huang2024multi,
  title={Multi-task Geometric Estimation of Depth and Surface Normal from Monocular 360 $\{$$\backslash$deg$\}$ Images},
  author={Huang, Kun and Zhang, Fang-Lue and Zhang, Fangfang and Lai, Yu-Kun and Rosin, Paul and Dodgson, Neil A},
  journal={arXiv preprint arXiv:2411.01749},
  year={2024}
}
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.

You are free to:

✅ Share — copy and redistribute the material in any medium or format

✅ Adapt — remix, transform, and build upon the material

Under the following terms:

🔗 Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

🚫 NonCommercial — You may not use the material for commercial purposes.

Note: For any commercial use or licensing inquiries, please contact the project maintainer.
