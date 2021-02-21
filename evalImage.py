import os, sys, argparse
import torch
import getPrediction, imageTransform


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
    probabilty_dist = getPrediction.prediction(transformed_image, CovidPredictionModel)

    # print for nn.LogSoftmax and nn.Softmax
    # terminal outputs
    # need to learn fancy formatting styles!
    print("")
    print("nn.LogSotmax Probabilty Distribution >>>")
    pLogSoftmax = torch.nn.LogSoftmax(dim=1)(probabilty_dist).tolist()[0]
    print("---------------")
    print(f"Normal: {pLogSoftmax[0]}", 
          f"Non-Covid Pneumonia: {pLogSoftmax[1]}",
          f"Covid Pneumonia: {pLogSoftmax[2]}", sep='\n')
    print("---------------")
    print("")
    print("")
    print("Sotmax Probabilty Distribution >>> ")
    pSoftmax = torch.nn.Softmax(dim=1)(probabilty_dist).tolist()[0]
    print("---------------")
    print(f"Normal: {pSoftmax[0]}", 
          f"Non-Covid Pneumonia: {pSoftmax[1]}",
          f"Covid Pneumonia: {pSoftmax[2]}", sep='\n')
    print("---------------")
    ### FIN :) ###