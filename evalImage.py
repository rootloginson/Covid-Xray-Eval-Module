import os, sys, argparse
import torch
import imageTransform 
import forwardPass
from outputFunction import outputFunc



parser = argparse.ArgumentParser(description='Pass a xray image path')
parser.add_argument('-img', type=str, required=True, help='path of the image')
args = parser.parse_args()

if __name__ == '__main__':

    ########## LOAD MODEL ##########
    try:
        # get working directory path
        path = os.getcwd()
        # load trained CNN model, (for CPU).
        CovidPredictionModel = torch.jit.load(path + '/ResultModelscriptmodule_CPU.pt')
    except Exception as e:
        print("Model could not be loaded. Check the path")
        print("exception type", e)
        sys.exit()
    ########## LOAD MODEL ##########

    print("\nEvaluating file", args.img, "...\n")
    # get transformed image
    transformed_image = imageTransform.makeInputforCNN(args.img)
    # get raw output for probability distribution
    logits = forwardPass.getLogits(transformed_image, CovidPredictionModel)
    print('If you use -i(interface) you can access to logit\'s object with the python object called "logits"\n')
    # print for nn.LogSoftmax and nn.Softmax
    # terminal outputs
    _ = outputFunc(logits, 'logsoftmax')
    _ = outputFunc(logits, 'softmax')
   
    ### FIN :) ###


### Example for debug mode ###
""" 
python -i evalImage.py
sm_output = torch.nn.Softmax(dim=1)(logits)
print(sm_output)
"""