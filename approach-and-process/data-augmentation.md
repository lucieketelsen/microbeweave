# Data Augmentation



 **Data Augmentation** is a technique that can be used to artificially expand the size of a training set by creating modified data from the existing one. It is a good practice to use **DA** if you want to prevent **overfitting**, or the initial dataset is too small to train on, or even if you want to squeeze better performance from your model.

 **Data Augmentation** is not only used to prevent **overfitting**. In general, having a large dataset is crucial for the performance of both **ML** and **Deep Learning** \(**DL**\) models. However, we can improve the performance of the model by augmenting the data we already have. It means that **Data Augmentation** is also good for enhancing the model’s performance.

I performed data augmentation in order to derive a dataset of comparable size to another dataset -- where both sets of equal size were required for a DCGAN, cycleGAN or for UGATIT. 

The following is the process I followed to derive a decent image dataset from a short time lapse microscopy capture:

Import openCV and libraries to open and operate on files

```text
import cv2
import os
import random
import shutil
```

Set link to video files

```text
videos = ['Gliding motility and cell organization of C. lytica CECT 8139 in an iridescent zone LN agar.avi',
          'Gliding motility and cell organization of C. lytica CECT 8139 in an iridescent zone MA medium.avi',
          'Gliding motility and cell organization of C. lytica CECT 8139 in an iridescent zone.avi']




train_dir = 'train/'
test_dir = 'test/'
if not os.path.isdir(train_dir):
  os.mkdir(train_dir)

if not os.path.isdir(test_dir):
  os.mkdir(test_dir)
```

Create a function to create quarters from each video frame 

```text
def crop(im):
  """
  function that splits images into two 4 quarters and resizes each
  quarter to 256x256
  """
  img_height , img_width , _ = im.shape
  quarters = []
  for i in range(2):
    for j in range(2):
      q = im[(int(img_height/4))*i:(int(img_height/4))*(i+1),(int(img_width/4))*j:(int(img_width/4))*(j+1),:]
      q = cv2.resize(q , (256,256),cv2.INTER_LINEAR)
      quarters.append(q)
  return quarters
```

Create a function to perform transformations on each quarter

```text
def augment(quarters):
  """
  function that augments each quarter frame into 7 different variations
  using rotate and flip, variations are:
  rot 90
  rot 180
  rot 270
  rot 0 + flip
  rot 90 + flip
  rot 180 + flip
  rot 270 + flip
  """
  aug_quarters = []
  for img in quarters:
    q = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    aug_quarters.append(q)
    q = cv2.rotate(img,cv2.ROTATE_180)
    aug_quarters.append(q)
    q = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    aug_quarters.append(q)
    q = cv2.flip(img,0)
    aug_quarters.append(q)
    q = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    q = cv2.flip(q,0)
    aug_quarters.append(q)
    q = cv2.rotate(img,cv2.ROTATE_180)
    q = cv2.flip(q,0)
    aug_quarters.append(q)
    q = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    q = cv2.flip(q,0)
    aug_quarters.append(q)
    
  return aug_quarters
```

Create a function to split the dataset into test and train

```text
def split_dataset(train_dir,test_dir,test_size):
  """
  function that takes (test_size) random instances from training set and
  moves them to given path
  """
  list = os.listdir(train_dir)
  no_files = len(list)
  count = 0
  n = random.sample(range(no_files + 1), test_size)
  for rand_no in n:
    shutil.move(train_dir + "frame%d.png" % rand_no ,test_dir + "frame%d.png" %count)
    count += 1
    
totalcount = 0  
test_size = 300
```

Loop through videos applying previously defined functions to crop, transform and split output into discrete datasets

```text
for video in videos:
  vidcap = cv2.VideoCapture(video)
  success,image = vidcap.read()
  while success:
    batch1 = crop(image)
    batch2 = augment(batch1)
    total_batch = batch1 + batch2
    for img in total_batch:
      cv2.imwrite(train_dir + "frame%d.png" % totalcount, img)     # save frame as PNG file
      totalcount += 1
    success,image = vidcap.read()
    

split_dataset(train_dir,test_dir,test_size)
```

