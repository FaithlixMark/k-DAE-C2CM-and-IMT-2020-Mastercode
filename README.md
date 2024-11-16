# k-DAE-C2CM-and-IMT-2020-Mastercode
K-AUTOENCODERS DEEP CLUSTERING
---------------------------------------------------------------
This repo contains the source code for our paper:

[**K-AUTOENCODERS DEEP CLUSTERING**](http://www.eng.biu.ac.il/goldbej/files/2020/02/ICASSP_2020_Yaniv.pdf) 
<br>
Yaniv Opochinsky, Shlomo E. Chazan, Sharon Gannot and Jacob Goldberger





---------------------------------------------------------------  
#### How to Use?
* clone the repo
* cd to the cloned dir 
* conda create -n k_dae python=3.6 
* conda activate k_dae 
* pip install - Requirements


### Run example: 
* `python main.py -dn mnist` 

#### Optional Args: 

* `--dataset_name` - The name of dataset [imt2020 / cost] default: imt2020
* `--save_dir` - path to output folder. (contains logs and model.)
* Dataset link: https://drive.google.com/drive/folders/1yZ6iIZC9cKYUyfTlUnZsYt2B1uotG2pd?usp=sharing

### Requirements
* Python = 3.11.2
* Tensorflow = 2.14.0
* Tensorflow-estimator = 2.14.0
* Numpy = 1.24.2
* Scipy = 1.10.1
* Scikit-learn = 1.3.2
* Keras = 2.14.0
* openpyxl = 3.1.2
* pandas = 2.2.0

For more info please refer to this paper: https://drive.google.com/file/d/1AMQVQDl2xmSlYwidBWxXGFyK3ef_qXhv/view?usp=sharing

### Abstract 
Channel modeling plays a significant role in designing and evaluating wire
less communication systems. Geometric-Based Stochastic Channel Modelling
 (GBSCM) rely on the statistical distribution of multipath waves which arrive at
 the receiver with similar parameters and are called clusters. This study investi
gates the performance of k-Deep Autoencoder (k-DAE) in clustering the multi
path components (MPC)s from the COST 2100 (C2CM) and International Mobile
 Telecommunications-2020 Channel Model (IMT-2020) channel models. Further
more, the algorithm is modified by its activation function and optimizer, namely:
 Sigmoid, Rectified Linear Unit (ReLU), with Adaptive Momentum (ADAM) and
 Stochastic Gradient Descent (SGD) as the optimizers. The results showed that the
 Sigmoid-SGD configuration outperforms all the other configurations, both in C2CM
 and IMT-2020 datasets. The configuration increased in Jaccard Index (JI) by 0.016
 at the 10th percentile, 0.019 for the median, and 0.3595 at the 90th percentile
 for the C2CM dataset compared to default configuration. On the other hand, the
 algorithm performance of the IMT-2020 data, increased in JI by 0.004, and 0.1, for
 the 10th, and 50th percentile, respectively.
