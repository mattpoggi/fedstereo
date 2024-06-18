
<h1 align="center"> Federated Online Adaptation for Deep Stereo <br> (CVPR 2024) </h1> 


<br>

:rotating_light: This repository contains the code for reproducing the main experiments of our work  "**Federated Online Adaptation for Deep Stereo**",  [CVPR 2024](https://cvpr2023.thecvf.com/)
 
by [Matteo Poggi](https://mattpoggi.github.io/) and 
[Fabio Tosi](https://fabiotosi92.github.io/),
University of Bologna


<div class="alert alert-info">


<h2 align="center"> 

[Project Page](https://fedstereo.github.io/) | [Paper](https://mattpoggi.github.io/assets/papers/poggi2024cvpr.pdf) |  [Supplementary](https://mattpoggi.github.io/assets/papers/poggi2024cvpr_supp.pdf) | [Poster](https://mattpoggi.github.io/assets/papers/poggi2024cvpr_poster.pdf) 
</h2>

<img src="https://fedstereo.github.io/images/teaser_fed.png" alt="Alt text" style="width: 800px;" title="architecture">

**Note**: 🚧 Kindly note that this repository is currently in the development phase. We are actively working to add and refine features and documentation. We apologize for any inconvenience caused by incomplete or missing elements and appreciate your patience as we work towards completion.

## :bookmark_tabs: Table of Contents

1. [Introduction](#clapper-introduction)
2. [Data Pre-processing](#file_cabinet-data-pre-processing)
    - [Download Pre-processed Data](#arrow_down-download-pre-processed-data)
    - [Prepare Data](#hammer_and_wrench-prepare-data)
3. [Pretrained Model](#inbox_tray-pretrained-model)
4. [Running the Code](#memo-running-the-code)
    - [Dependencies](#hammer_and_wrench-dependencies)
    - [Setting up Config Files](#hammer_and_wrench-setting-up-config-files)
    - [Testing](#rocket-testing)
5. [Qualitative Results](#art-qualitative-results)
6. [Contacts](#envelope-contacts)
7. [Acknowledgements](#pray-acknowledgements)

</div>

## :clapper: Introduction

<h4 align="center">

</h4>

<img src="https://fedstereo.github.io/images/FED-method.png" alt="Alt text" style="width: 800px;" title="architecture">


We introduce a novel approach for adapting deep stereo networks in a **collaborative manner**. By building over principles of **federated learning**, we develop a distributed framework allowing for demanding the optimization process to a number of clients deployed in different environments. This makes it possible, for a deep stereo network running on **resourced-constrained devices**, to capitalize on the adaptation process carried out by other instances of the same architecture, and thus improve its accuracy in challenging environments even when it cannot carry out adaptation on its own. Experimental results show how **federated adaptation** performs equivalently to on-device adaptation, and even better when dealing with **challenging environments**.


:fountain_pen: If you find this code useful in your research, please cite:

```bibtex
@inproceedings{Poggi_2024_CVPR,
    author    = {Poggi, Matteo and Tosi, Fabio},
    title     = {Federated Online Adaptation for Deep Stereo},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024}
}
```

## :movie_camera: Watch Our Presentation Video!

<a href="https://www.youtube.com/watch?v=gVpWsjrUTJc">
  <img src="https://fedstereo.github.io/images/youtube.png" alt="Watch the video" width="800">
</a>

## :file_cabinet: Data Pre-processing

We pre-process the different datasets and organize them in ``.tar`` archives. For the sake of storage space, left and right images are converted in ``jpg`` format.
You have several options for getting the data to run our code. 

## :arrow_down: Download Pre-processed data

**(Recommended)** If you wish to download pre-processed `tar` archives, drop us an [email](#envelope-contacts) and we will share access to all pre-processed archives stored in our OneDrive account (unfortunately, we cannot share public links).
As we might need a while to answer, you can pre-process the data on your own (see below).

## :hammer_and_wrench: Prepare Data

Most data required for pre-processing is hosted on Google Drive. To download them, please install `gdown`

``pip install gdown==4.7.3``

For KITTI sequences, please `imagemagick` for converting pngs to jpgs, as well as `opencv-python`

``sudo apt install imagemagick``

``pip install opencv-python==4.8.1``

The scripts for preparing data will download several zip files, unpack them and move files all around.
It will take "a while", so just go around and let them cook :)

Missing some of the above dependencies will cause the following scripts to fail.

### :arrow_down: Download Demo Data (DSEC)

You can download from Google Drive the DSEC sequences used in our experiments with the following script:

``bash prepare_data/download_preprocessed_dsec.sh``

It will directly store `tar` archives under `sequences/dsec` folder (this requires ~14 GB storage).

Other sequences are too large to host on Google Drive, so we provide scripts to download them from the original sources and process them locally (see below).

### :arrow_down: Prepare DrivingStereo Data

Run the following script:

``bash prepare_data/prepare_drivingstereo.sh > /dev/null``

This will download the left, right and groundtruth folders from the [DrivingStereo](https://drivingstereo-dataset.github.io/) website, as well as our pre-computed proxy labels from [Google Drive](https://drive.google.com/file/d/13OSlqnHWBxIOk_aJya3sOfAwhxQTKE0p/view?usp=drive_link). 
Then, it will prepare `tar` archives and will store them under `sequences/drivingstereo/` folder (this requires >45 GB storage).

### :arrow_down: Prepare KITTI Data

Run the following script:

``bash prepare_data/prepare_kitti_raw.sh > /dev/null``

This will download the left, right and groundtruth depth folders from [KITTI]() website, as well as our pre-computed proxy labels from [Google Drive](https://drive.google.com/file/d/1t9X12cYAQqJ6G8U2-XzO4oca3u2KnsYl/view?usp=sharing). 
Then, it will convert ``png`` color images into ``jpg`` and groundtruth depth maps into disparity maps, and will prepare ``tar`` archives and store them under `sequences/kitti_raw/`` folder (this requires >25 GB storage)

## :inbox_tray: Pretrained Model

Download pretrained weights by running:

`bash prepare_data/download_madnet2_weights.sh`

This will store `madnet2.tar` checkpoint under `weights` folder.

At now, we do not plan to release the code for pre-training. If you are interested in training MADNet 2 from scratch, you can insert it in [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) pipeline and use the `training_loss` defined in `madnet2.py`. Then replace `AdamW` with `Adam` and use the learning rate and scheduling detailed in the supplementary material.

## :memo: Running the Code

Before running experiments with our code, you need to setup a few dependencies and configuration files.

### :hammer_and_wrench: Hardware setup

Federated experiments require up to 4 GPUs (one per client), while one GPU is sufficient for single-domain experiments.

### :hammer_and_wrench: Dependencies

Our code has been tested with CUDA drivers ``470.239.06``, ``torch==1.12.1+cu113``, ``torchvision==0.13.1+cu113``, ``opt_einsum==3.3.0``, ``opencv-python==4.8.1``, ``numpy==1.21.6`` and ``webdataset==0.2.60``.

### :hammer_and_wrench: Setting up Config Files

Clients and Server's behaviors are defined in ``.ini`` config files.
Please refer to ``cfg/README.md`` for detailed instructions

### :rocket: Testing

``python run.py --nodelist cfgs/multiple_clients.ini --server cfg/server.ini``

Arguments:

``--nodelist``: list of config files for client(s) running during the experiments

``--server``: config file for the server to be used in federated experiments

``--verbose``: prints stats for any client running (if disabled, only the listening client will print stats)

``--seed``: RNG seed

Please note that the performance of federated adaptation may change from run to run and depends on your hardware (e.g., after refactoring, we tested the code on a different machine and bad3 in most cases improved by 0.10-0.20% roughly)

To run a single client:

``python run.py --nodelist cfgs/single_client.ini``

## :art: Qualitative Results

In this section, we present illustrative examples that demonstrate the effectiveness of our proposal.

**KITTI - Residential sequence**
<img src="https://fedstereo.github.io/images/qualitatives_residential.PNG" alt="GIF" width="800" >
</p>
 
**DrivingStereo - Rainy sequence**
<img src="https://fedstereo.github.io/images/qualitatives_rainy.PNG" alt="GIF" width="800" >
</p>

**DSEC - Night#4 sequence**
<img src="https://fedstereo.github.io/images/qualitatives_night4.PNG" alt="GIF" width="800" >
</p>


## :envelope: Contacts

For questions, please send an email to m.poggi@unibo.it or fabio.tosi5@unibo.it

## :pray: Acknowledgements

We would like to extend our sincere appreciation to the authors of the following projects for making their code available, which we have utilized in our work:

- We would like to thank the authors of [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), [CREStereo-PyTorch](https://github.com/ibaiGorordo/CREStereo-Pytorch), [IGEV-Stereo](https://github.com/gangweiX/IGEV/tree/main/IGEV-Stereo), [UniMatch](https://github.com/autonomousvision/unimatch), [CoEX](https://github.com/antabangun/coex), [PyTorch-HITNet](https://github.com/MJITG/PyTorch-HITNet-Hierarchical-Iterative-Tile-Refinement-Network-for-Real-time-Stereo-Matching) and [TemporalStereo](https://github.com/youmi-zym/TemporalStereo) for providing their code, which has been instrumental in our experiments.
