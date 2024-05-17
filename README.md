# PSPNet for lane segmentation
## PSPNet architecture
PSPNet is stand for **Pyramid Scene Parsing Network**
### Receptive feild
The Receptive Field (RF) in deep learning is defined as the size of the area in the input that creates the feature. 
It is essentially a measure of the relationship of an output feature (of any layer) with the input area (patch).

<br>
<img src="https://github.com/ngxx-fus/MLAI_FINAL_EXAM/assets/75427876/7b587f7c-31ab-4a51-adc1-d6a2a4392f7a"  width="401" text-align: center>
<br>

### Pyramid Pooling Module (PPM)
PPM is added to increase receptive field. Feature-maps will convolve with many kernels of different sizes.

<br>
<img src="https://github.com/ngxx-fus/MLAI_FINAL_EXAM/assets/75427876/c54b34e0-5318-4b62-8bd2-e7de60f4682e" width="802">
<br>

### Auxiliary Loss 
Reducing effect from Vanishing Gradient Descent by computing the loss after res4b22 residue block.

<br>
<img src="https://github.com/ngxx-fus/MLAI_FINAL_EXAM/assets/75427876/45aaebf5-7a8b-4bb3-bf20-9230d0412736" width="401">
<br>


## Dataset
Note: The img and mask have the same filename (include extension)!
In this project:
- input img size (wxh): 2048x1024
- training img size (wxh): 401x401
- no. of training img : 415 images
- no. of testing img  : 20 images
- img source: [cityscape](https://paperswithcode.com/dataset/cityscapes)


      root_dir:
      + Default_Mask:
      |    - null_img.png
      |
      + Model_Weights:
      |    -  resnet50_v2.pth 
      |    -  29th_modelPSPNet.pth 
  
      dataset_dir:
      + IMG
      |    -  img_01.png
      |    -  img_02.png
      |    -  img_##.png
      + MASK
      |    -  img_01.png
      |    -  img_02.png
      |    -  img_##.png
      - trainval.txt
      - test.txt


## Labels
The model was built for 21 classes (Labels), but in this project we only use **four**.
| ID |    LABEL's NAME   | 
| -- | :----------------|
| 0  | VOID              |
| 1  | DUONG_DI          |
| 2  | LAN_HIEN_TAI      |
| 3  | LAN_TRAI_0        |
| 4  | LAN_PHAI_0        |
| 5  | VOID              |
| ...| VOID              |
| 20 | VOID              |


## The result 
### In train_dataset:
> - Train loss = 0.05682641424238682
> - Accuracy = 0.9867874002456665
> - IoU = 0.9490965843200684
> - Dice = 0.9733260941505432
### Review 

<br>
<img src="https://github.com/ngxx-fus/MLAI_FINAL_EXAM/assets/75427876/736a4240-5003-4230-893d-bd0550a397de" width="512">
<br>

<br>
<img src="https://github.com/ngxx-fus/MLAI_FINAL_EXAM/assets/75427876/283d1975-058a-4acd-92b6-0b5dcc62070a" width="512">
<br>

<br>
<img src="https://github.com/ngxx-fus/MLAI_FINAL_EXAM/assets/75427876/155daa8d-4b84-40ff-8c44-8689daf22793" width="512">
<br>

<br>
<img src="https://github.com/ngxx-fus/MLAI_FINAL_EXAM/assets/75427876/b7995c8e-b81d-4477-bb0c-18f0f597b51c" width="512">
<br>

<br>
<img src="https://github.com/ngxx-fus/MLAI_FINAL_EXAM/assets/75427876/fa1ae039-830b-4231-b1e0-c8b9926e76ca" width="512">
<br>

<br>
<img src="https://github.com/ngxx-fus/MLAI_FINAL_EXAM/assets/75427876/fab02b1b-f45f-4c64-b5a0-318553630f81" width="512">
<br>

<br>
<img src="https://github.com/ngxx-fus/MLAI_FINAL_EXAM/assets/75427876/d490c80b-826d-42d2-9379-badd072f04bb" width="512">
<br>

<br>
<img src="https://github.com/ngxx-fus/MLAI_FINAL_EXAM/assets/75427876/26d83137-d0b5-4ded-883e-29d8c95548ef" width="512">
<br>

