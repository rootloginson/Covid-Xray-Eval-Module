# Covid-Xray-Eval-Module

Kaggle DL project with PyTorch in a nutshell. Python modules to evaluate X-ray images to check for covid caused lung inflammation. 
This repo is the summary of my very first 0 to End project in ComputerScience and DeepLearning. Feel free to comment.

Image transform + Model + Prediction / All-in-one

---
### How to use ? 
Open a terminal in the same directory as the python files. 
 > git clone https://github.com/rootloginson/Covid-Xray-Eval-Module.git

 > python evalImage.py -img 'imagename'
  
***-img***  is for argparser to pass image name. 

if image is in the same directory as the python files pass only the name. If not, pass 'directory/imagename'. 

Note: I  letter in eval image look like l  :) . 

---

**!!!** Medical image datasets are hard to find. It is very hard to collect. Most of the X-Ray images on the internet belong to Covid kaggle datasets. Therefore, The X-ray image that you will try, may already be in the dataset that I have used for training and validations. 
I hope someday world will get through this problem.

---

**To run the test file**
 > python -m unittest

---

**evalimage.py** ->> main module. Loads the model. Pass image name and model to below functions as an argument.

**imageTransform.py** ->> applies transforms that is applied to input image of the PyTorch validation and test model.

```python
def makeInputforCNN(xray_image_name):
"""
  Arguments: 
    xray_image_name (str):
        if image file is in the same directory as imageTransform.py pass the imagename.
        function adds the image name to working directory adress.
  Returns:
    transformed image (torch.tensor):
        transformed image tensor. This will be the input of trained model for forward pass.
        This return object will send to get_prediction as an argument, along with the loaded model.
"""
```

**forwardPass.py** ->> uses the return object of imageTransform.py to forward pass on loaded model
```python
def getLogits(img_tensor, CovidPredictionModel):
    """
    Argument:
        img_tensor (torch.tensor): return object of makeInputforCNN function.        
        CovidPredictionModel: pytorch trained model for CPU.
    Returns:
      forward pass output (torch tensor):
        !! No softmax function applied. This will be applied in evalImage.py to print
        LogSoftmax and Softmax output. In model nn.CrossEntropyLoss() has been used. 
        !!! nn.CrossEntropyLoss combines nn.LogSoftmax + nn.NLLLoss
    """
```
