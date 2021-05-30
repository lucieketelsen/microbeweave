# model and solution

## models

Numerous attempts were made to create a model suitable to the learning problem. Not all were successful however attempts at setting up environments and implementations were valuable learning experiences. 

The attempts were:

* u-gat-it
* cycleGAN
* GAN \(Keras\)
* VAE
* transfer learning with keras
* deeptrack\(v2\)

### UGATIT

U-GAT-IT is short for Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation. The official source code can be found at [https://github.com/taki0112/UGATIT](https://github.com/taki0112/UGATIT)

UGATIT is the work of ****Junho Kim, Minjae Kim, Hyeonwoo Kang, Kwanghee Lee. Their abstract explains the approach they explored with UGATIT: 

> _We propose a novel method for unsupervised image-to-image translation, which incorporates a new attention module and a new learnable normalization function in an end-to-end manner. The attention module guides our model to focus on more important regions distinguishing between source and target domains based on the attention map obtained by the auxiliary classifier. Unlike previous attention-based methods which cannot handle the geometric changes between domains, our model can translate both images requiring holistic changes and images requiring large shape changes. Moreover, our new AdaLIN \(Adaptive Layer-Instance Normalization\) function helps our attention-guided model to flexibly control the amount of change in shape and texture by learned parameters depending on datasets. Experimental results show the superiority of the proposed method compared to the existing state-of-the-art models with a fixed network architecture and hyper-parameters._

Requirements:

* clone the repository 
* python 3.6
* tensorflow 1.14
* anaconda is recommended
* cuda GPU is required \(tensorflow-gpu will be installed as part of dependencies\)
* docker is also recommended though not required

_If running locally , the following commands can be run:_



```bash
# Anaconda
conda env create -f environment.yml
conda activate UGATIT

# Pip / Python3.5
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

Setting up an environment.yml helps with this process, which can be structured as follows:

```bash
name: UGATIT
dependencies:
  - python=3.5
  - matplotlib
  - numpy
  - pip
  - pip:
    - opencv-python
    - Pillow
#    - tensorflow==1.15.0 # CPU support
    - tensorflow-gpu==1.15.0
```

Alternatively, UGATIT can be run in the cloud using Google Colab, AWS etc. 

To set up AWS EC2:

* sign in/sign up
* request service limit
* create an ssh key and upload it to aws
* launch an instance \(must include cuda and gpu, per the project dependencies\)
* connect to your instance
* set up server

\(don't forget to stop your instance when complete\)

Collecting and preparing datasets for use is the most difficult part of this project_. Data must be set up according to the structure below:_ 

```bash
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.png
           └── ...
```

A well known use of UGATIT is the problematic though technically successful **selfie2anime.** The structure of this dataset gives an idea of what is required to give decent results using UGATIT. 

* selfie - The size of the training dataset is 3400, and that of the test dataset is 100, with the image size of 256 x 256.
* anime - 3400 and 100 for training and test data respectively

 **Images need to be cropped** to an appropriate size before using them. This can be done by editing`resize.py` and ensuring the DATASET\_NAME matches yours.

```bash
paths = [
    "dataset/DATASET_NAME/testA/"
    "dataset/DATASET_NAME/testB/"
    "dataset/DATASET_NAME/trainA/"
    "dataset/DATASET_NAME/trainB/"
]
```

Once images have been cropped, named and placed in all four folders, they can be resized down to 256x256. 

```
python resize.py
```

Different parameters that can be passed in. They can be found at the top of `main.py`; some important ones are below:

```bash
parser.add_argument('--phase',      default='train',        help='[train / test]')
parser.add_argument('--light',      default=False,          help='[full version / light version]')
parser.add_argument('--dataset',    default='selfie2anime', help='dataset_name')
parser.add_argument('--epoch',      default=100,            help='The number of epochs to run')
parser.add_argument('--iteration',  default=10000,          help='The number of training iterations')
parser.add_argument('--batch_size', default=1,              help='The size of batch size')
parser.add_argument('--print_freq', default=1000,           help='The number of image_print_freq')
parser.add_argument('--save_freq',  default=1000,           help='The number of ckpt_save_freq')
parser.add_argument('--decay_flag', default=True,           help='The decay_flag')
parser.add_argument('--decay_epoch',default=50,             help='decay epoch')
```

To train, run:

```bash
python main.py --dataset YOUR_DATASET_NAME
```

Training begins shortly after running. You should see an output line the following indicating that the process is running.

```bash

```

 After each 100 iterations, UGATIT will generate some samples that can be viewed in the `samples` folder. This allows you to view progress while the training occurs.

With these settings, a checkpoint is created every 1000 iterations also. A checkpoint is a state that inference can be performed on, or resumed from.

If you need to resume training, or potentially you'd like to try transfer learning and continuing to train an existing model; then the process of resuming training is very simple.

 Re-run the training command with the same parameters as before. You'll notice that if you change parameters then the folder name within the `checkpoint`, `samples`, `results` and `logs` folder all change.

To test, run:

```bash
python main.py --dataset YOUR_DATASET_NAME --phase test --light True
```

To make a test video, make sure to set your video device in the `State` class of `main.py` if you have a unique setup.

By default it'll use the first video device attached

```bash
python main.py --dataset YOUR_DATASET_NAME --phase video --light True
```

{% hint style="info" %}
 Super-powers are granted randomly so please submit an issue if you're not happy with yours.
{% endhint %}

Once you're strong enough, save the world:

{% code title="hello.sh" %}
```bash
# Ain't no code for that yet, sorry
echo 'You got to trust me on this, I saved the world'
```
{% endcode %}



