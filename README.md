# Shape Coded Particles (SCP)
This repository contains the code, experiments, and examples from our work titled "Deep Learning-Based Recognition of Shape-Coded Microparticles". The experiments were conducted to test and demonstrate the usability of semantic segmentation models for our particle shapes. Given the nature of our dataset ("one class per image"), we implemented Augmentation by Translocation (ABT) to compensate for the limitations of our small dataset and enhance learning results.
- [Examples](#Examples)
- [Introduction](#Introduction)
- [Databases](#Database)
- [Results](#Results)
- [Future Work](#Future-Work)
- [Future Research](#Future-Research)
- [Citation](#Citation)
- [Contact](#Contact)

## Quick start  
TBD;

## Examples
### Example: Augmentation by Translocation (ABT) general
An example implementation of the abt algorithm can be found in the ABT folder ([abt_general.py](ABT/abt_general.py)).  
This example showcases the abt algorithm on a simple generated dataset of 2 classes (no need to download a dataset).  
[abt_general.py](ABT/abt_general.py) is the best place to start if you want to use the abt algorithm on your own dataset.

### Example: Augmentation by Translocation (ABT) for SCP
A simple implementation of the ABT algorithm can be found in the ABT folder ([abt_for_scp.py](ABT/abt_for_scp.py)).  
The example is written for the 10S_raw_abt dataset.  
10S_raw_abt can be found [here](https://drive.google.com/file/d/1IdNliHuYhy35FoiNLBzOWP3JlsupTaol/view?usp=share_link).

1. Download the 10S_raw_abt.zip file.
2. Unzip the file and adjust the path in the main section of the abt_for_scp.py file.

### Example: Semantic Segmentation for SCP 
[particle_segmentation.ipynb](particle_segmentation.ipynb) is a notebook showcasing examples for training different models for multiclass segmentation.  
The corresponding datasets can be downloaded [here](https://drive.google.com/file/d/1IdNliHuYhy35FoiNLBzOWP3JlsupTaol/view?usp=share_link).  

1. Download one of the dataset .zip files.  
2. Start the notebook with GPU support (e.g. Google Colab).
3. Upload the dataset .zip file to the notebook.
4. Follow the steps of the notebook.

## Introduction
Encoded particles have been utilized for multiplexed diagnostics, drug testing, and anti-counterfeiting applications. Recently, shape-coded hydrogel particles with amphiphilic properties have enabled amplified duplexed bioassays. However, a limitation in reading multiple particle shape codes in an automated and time-efficient manner hinders the widespread adoption of such powerful diagnostic platforms. In this work, we applied established deep learning-based multi-class segmentation models, such as U-Net, Attention U-Net, and UNet3+, to detect five or more particle shape codes within a single image in seconds.

We demonstrated that the models tested provided average results when implemented on an imbalanced and limited raw dataset, with the best intersection-over-union (IoU) scores being 0.76 and 0.46 for six- and eleven-class segmentation, respectively. We introduced the Augmentation by Translocation (ABT) technique, significantly enhancing the performances of the tested models, with the best IoU scores for the six and eleven classes increasing to 0.92 and 0.74, respectively. These initial findings underscore the potential of shape-coded particles in multiplexed bioassays.

![SCP - shapes](doc/Table%201%20-%20shape%20information%20-%20low.jpg)


For the deep learning library, we used [keras-unet-collection](https://github.com/yingkaisha/keras-unet-collection) and [segmentation-models](https://github.com/qubvel/segmentation_models).


## Database
### Annotation
We annotated our dataset with [MiVOS](https://github.com/hkchengrex/MiVOS), a tool for interactive annotation of video and image data.

### Raw Database
The raw dataset comprises 10 different shapes, each represented by around 30 images along with their corresponding instant segmentation masks in MS-COCO format.
| Shape:   | SC0 | SC1 | SC2 | SC3 | SC13 | SC4 | 4H | N  | TT | LL  | BG   |
|----------|-----|-----|-----|-----|------|-----|----|----|----|-----|------|
| Images:  | 35  | 34  | 28  | 37  | 32   | 32  | 28 | 24 | 37 | 38  | 325  |
| Instances| 128 | 147 | 102 | 128 | 128  | 116 | 128| 116| 148| 165 | 325  |
| Pixel:   | 80k | 90k | 61k | 77k | 83k  | 73k | 84k| 63k| 105k| 110k| 20468k|
| 10S11C:  | *  | *  | *  | *  | *   | *  | * | * | * | *  | +    |
| 5S6C:    | *  | -  | -  | -  | -   | *  | * | - | * | *  | +   |

### Raw Database for ABT
The raw dataset for ABT is processed for compatibility with the ABT algorithm. Each class is represented in a folder containing all the single instance masks.  
### 5S Database
The 5S database is a subset of the raw database, containing only five classes and processed to multiclass segmentation masks. It also includes basic augmentation (BA) and ABT augmentation of the dataset.
### 10S Database
The 10S database incorporates all the images from the raw database and is processed to multiclass segmentation masks. It also features both basic augmentation (BA) and ABT augmentation of the dataset.



## Results
### Raw, BA, and ABT Image examples
![Raw, BA, ABT](doc/Figure%202%20-%20Augmentation%20examples.png)
### Segmentation results for the 5S Dataset
![Samantic segmentation results](doc/Figure%203%20-%205S%20-%20low.png)

### Results for the 10S Dataset with a fixed training loop
| Model    | IoU Score(0.5)       | Dice Score          | AP(0.5)               | AR                  |
|----------|----------------------|---------------------|-----------------------|---------------------|
|     | Raw - BA - ABT       |  Raw - BA - ABT         |  Raw - BA - ABT            |  Raw - BA - ABT                 |
| UNet     | 0.16 - 0.16 - 0.54   | 0.14 - 0.11 - 0.49  | 0.51 - 0.51 - 0.80    | 0.50 - 0.50 - 0.88  |
| **A-UNet** | 0.18 - 0.18 - **0.74** | 0.12 - 0.12 - **0.59** | 0.51 - 0.51 - **0.91** | 0.50 - 0.50 - **0.91** |
| UNet3+   | 0.46 - 0.60 - 0.72   | 0.34 - 0.42 - 0.53  | 0.72 - 0.82 - 0.87    | 0.81 - 0.89 - 0.91  |

## Future Work
Since this is a very simple implementation of ABT, there are many ways to improve the algorithm.
One easy approach would be to optimize the location where the particles will be placed.
Right now it is random which can result in intensity differences between the particles and the background. 
Another simple approach would be to use nonrigid BA as a base for ABT to get more variance.


## Future Research
We tested the ABT algorithm only on our SCP datasets where the actual location of the particles has no relation to class characteristics.

Since SCP have similar optics to other microscopic medical images of cells, ABT could be used to generate new data for those datasets.

## Citation
TBD;

## Contact
If you have any questions or suggestions, feel free to contact us at:  
ghulam.destgeer@tum.de, akif.sahin@tum.de, leander.eijnden@tum.de 