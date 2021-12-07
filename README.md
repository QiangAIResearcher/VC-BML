## VC-BML

This is the code repository of the NeurIPS 2021 paper [Variational Continual Bayesian Meta-Learning](https://openreview.net/forum?id=VH2og5jlrzm). 

VC-BML is a meta-learning model that deal with online streaming tasks from a non-stationary distribution. It is a fully Bayesian model where both meta-parameters and task-specific parameters are latent random variables. Specifically, the meta-parameters follow a Dynamic Gaussian Mixture model, with the number of component distributions determined by a Chinese Restaurant Process. The posteriors of these latent variables are inferred with structured variational inference. 



## Requirements 

+ Python 3.7.9
+ pytorch 1.7.1 
+ numpy 
+ pandas
+ tqdm



## Data Setup

+ In ``` config/seqdataset.json```, specify the location of your custom ```$DATASET_PATH ```. 
+ Download the datasets ([Omniglot](https://github.com/brendenlake/omniglot/tree/master/python), [CIFAR-FS](https://github.com/bertinetto/r2d2), [*mini*-Imagenet](https://github.com/renmengye/few-shot-ssl-public), [VGG-Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)) and unzip these datasets in your custom ```$DATASET_PATH```. The code assumes the following structure:

```
$DATASET_PATH
├── omniglot_raw
|   ...
├── cifar100_raw
|   ...
├── mini_imagenet_raw 
|   ...
├── vggflowers_raw 
|   ...
```



+ After downloading the datasets, run the data preparation scripts in the data_generate folder. For example, run the following command to prepare Omniglot for training:

```bash
python data_generate/prepare_omniglot.py
```



## Training

Run the training script to train VC-BML:

```
python traininig/main.py --config_path config/seqdataset.json
```

Note that details of hyperparameters can be found in the configuration file "seqdataset.json".  



## Citation

If you find our code useful, please kindly cite the paper:

```
@article{zhang2021variational,
  title={Variational Continual Bayesian Meta-Learning},
  author={Zhang, Qiang and 
  		  Fang, Jinyuan and 
  		  Meng, Zaiqiao and 
  		  Liang, Shangsong and 
  		  Yilmaz, Emine},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

