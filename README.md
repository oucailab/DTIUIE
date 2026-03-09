# Downstream Task Inspired Underwater Image Enhancement: A Perception-Aware Study from Dataset Construction to Network Design

This repository contains the official implementation of the following paper:
> **Downstream Task Inspired Underwater Image Enhancement: A Perception-Aware Study from Dataset Construction to Network Design**<br>
> Bosen Lin, Feng Gao<sup>*</sup>, Yanwei Yu, Junyu Dong, Qian Du <br>
> IEEE Transactions on Image processing, 2025<br>

[[Paper](https://arxiv.org/abs/2603.01767)]

## Dependencies and Installation
1. Clone Repo
    ```bash
    git clone https://github.com/oucailab/DTIUIE.git
    cd DTIUIE
    ```

2. Create Conda Environment
    ```bash
    conda env create -f environment.yaml
    conda activate dtiuie
    ```

## Get Started
### Prepare pretrained models & dataset 

1. You are supposed to download our pretrained model first in the links below and put them in dir `./checkpoints/`:

<table>
<thead>
<tr>
    <th>Model</th>
    <th>:link: Download Links </th>
</tr>
</thead>
<tbody>
<tr>
    <td>DTIUIE</td>
    <th>[<a href="https://pan.baidu.com/s/1-zdVG12eH3l7mxlnmdePRg?pwd=tsgi ">Baidu Disk (pwd: tsgi)</a>] </th>
</tr>
</tbody>
</table>

2. TI-UIED Dataset used in our work can be downloaded in the links below:
TI-UIED: [<a href="">Google Drive (TBD)</a>] [<a href="">Baidu Disk (TBD)</a>]

Unzip the TI-UIED dataset and put in dir `./data/`.
```bash
cat TIUIED_* > TIUIED.tar.gz
tar xvzf TIUIED.tar.gz
```


**The directory structure will be arranged as**:
```
checkpoints
    |- DTIUIE_ckpt.pth
dataset
    |- test
        |- images
            |- ***.jpg
            |- ...
        |- masks_png
            |- ***.png
            |- ...
        |- reference
            |- ***.jpg
            |- ...
    |- train
        |- images
            |- ***.jpg
            |- ...
        |- masks_png
            |- ***.png
            |- ...
        |- reference
            |- ***.jpg
            |- ...
```

### Training & Testing
Run the following commands for training:

```bash
python underwater_train.py --model model_dtiuie --database TIUIED
```

Run the following commands for testing:
```bash
python underwater_test.py --model model_dtiuie --draw_images
```

## Citation
If you find our repo useful for your research, please cite us:
```
@article{lin2026dtiuie,
  title={Perception-Downstream Task Inspired Underwater Image Enhancement: A Perception-Aware Study from Dataset Construction to Network Design},
  author={Bosen Lin, Feng Gao, Yanwei Yu, Junyu Dong, Qian Du},
  journal={IEEE Transactions on Image processing},
  year={2026}
}
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.

