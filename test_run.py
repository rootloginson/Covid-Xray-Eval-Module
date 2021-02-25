import unittest
import imageTransform
import forwardPass
import os
import torch
from PIL import Image

# current working directory 
c_dir = os.getcwd()
# test image path (3x500x500 white image)
img_path = 'TestFiles/whiteRGB500x500.jpeg'

# compare expected output 
class TestImageTransfrom(unittest.TestCase):
    
    def test_ImageTransform(self):
        # open image
        img = Image.open(c_dir+'/'+img_path)
        # apply transform
        transformed_img = imageTransform.apply_transforms(img)
        # load precalcualted output tensor
        comparison_tensor = torch.load(c_dir+'/TestFiles/apply_transformed_img.pt')
        
        # make comparison
        msg = 'ImageTransform file or reference iamge has changed'
        comparison_result = torch.allclose(transformed_img, comparison_tensor)
        self.assertEqual(comparison_result, torch.tensor(True))


    def test_checkSizeOfInput(self):
        # test shape of input for CNN forward 
        comparison_tensor = torch.zeros(1,3,224,224).shape
        transformed_img = imageTransform.makeInputforCNN(c_dir+'/'+img_path).shape
        
        # make comparison
        msg = 'Dimensions are wrong'
        self.assertEqual(transformed_img, comparison_tensor, msg)


    def test_forwardPass(self):
        # note: I have moved this here from forwardPass module. When forwardPass module directly,
        # it was making its own test. Below code adapted from there!  

        ########## LOAD MODEL ##########
        try:
            # load trained CNN model, (for CPU).
            CovidPredictionModel = torch.jit.load(c_dir + '/ResultModelscriptmodule_CPU.pt')
        except Exception as e:
            print("Model could not be loaded. Check the path")
            print("exception type", e)
            sys.exit()
        ################################

        output = forwardPass.getLogits(torch.ones(1, 3, 224, 224), CovidPredictionModel)
      
        # compare expected output 
        test_output = torch.load(c_dir + '/TestFiles/test_logits.pt')
        assert torch.allclose(output, test_output) == torch.tensor(True), 'Values do not match.'


if __name__ == '__main__':
	unittest.main()