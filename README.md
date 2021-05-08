# Multi-Dimension Model Training n PyTorch
<img src="https://user-images.githubusercontent.com/59862546/117540979-59b64100-b02f-11eb-9ea9-457ecf2e2271.png" width="400" height="300"> <img src="https://user-images.githubusercontent.com/59862546/117541029-9124ed80-b02f-11eb-91a0-4e5f4f0f062a.png" width="400" height="300">

## Dataset
In this project, model build for multi dimension images like image dimension >3(rgb). so data created in such a manner where images are random 8 dimension images with their respective random labels.
```
   self.image = np.random.rand(5000,224,224,8)
   self.labels = np.random.choice([0, 1], size=(5000,), p=[0.6,0.4])
```

## Image classification using pretrained model on random multi dimension image data
below are the prettrained model used for this problem:
1. resnet18
2. vgg16
3. densenet161
4. alexnet

## prediction
```python
import torch
from utils.utils import *
x,y = dataset
model = torch.load('model_multi_dim.pth')
y_pred = model(x)
accuracy = binary_acc(y_pred,y)
```

# Give a :star: To This Repository!
