# CFNet
Implementation from paper "Co-Occurrent Features in Semantic Segmentation"

![CFnet](https://user-images.githubusercontent.com/55435653/179143849-50dc8800-1087-4a8f-9212-fe89f0cab37b.png)


## Paper
- Link: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Co-Occurrent_Features_in_Semantic_Segmentation_CVPR_2019_paper.pdf

## Main Library Features
- High Level API
- ResNet101 backbones
- Models can be used as Subclassed or Functional Model
- Implement like paper code and adding multi head attention to CFNet with source code from TASM

## Installation and Setup
**Requirements**
**Windows or Linus**
1) Python 3.6 or higher
2) tensorflow >= 2.3.0 (>= 2.0.0 is sufficient if no efficientnet backbone is used)
3) numpy

**Clone Repository**

    $ git clone https://github.com/huy105/CFNet.git

or directly install it:<br>
**Pip Install Repository**

    $ pip install git+https://github.com/huy105/CFNet.git
    
## Training Pipeline

Please check that **Tensorflow** is installed on your computer.

To import the model just use the standard python import statement

Firstly, import function to get base model (backbone) and main head (CFNet):

```python
from backbone import create_base_model
from main import CFNet
```
Then, get the backbone model, output layers, height and width:

```python
BACKBONE_NAME = "resnet101"
WEIGHTS = "imagenet"
HEIGHT = 224
WIDTH = 224
```

```python
model, layers_outputs, layer_names = create_base_model(name= BACKBONE_NAME, weights= WEIGHTS, height= HEIGHT, width= WIDTH, channels=3)
```
After that, create full model (base model + head) with parameter you want

```python
CFnet_model = CFNet(n_classes = 5, base_model = model, output_layers = layers_outputs, n_heads=2, n_mix = 4,backbone_trainable = True)
```

If you want to use the Functional Model class define instead:

```python
CFnet_model = CFNet(n_classes = 5, base_model = model, output_layers = layers_outputs, n_heads=2, 
n_mix = 4,backbone_trainable = True,  height= HEIGHT, width= WIDTH).model()
```

## References
<p>Using custom layers and main code from TensorFlow Advanced Segmentation Models (TASM).</p>

- TensorFlow Advanced Segmentation Models, GitHub, GitHub Repository, https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models
- Code from author of paper, GitHub, GitHub Repository, https://github.com/zhanghang1989/PyTorch-Encoding





