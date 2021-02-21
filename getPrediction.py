import torch
import os


def prediction(img_tensor, CovidPredictionModel):
    """
    Argument:
        transformed image tensor: torch.tensor

        Return object of "def makeInputforCNN(img_path)" function in
        imageTransform.py module.
    """
    # output is nn.CrossEntropyLoss
    probability = CovidPredictionModel(img_tensor)
    # return raw output
    return probability


# debug test purposes, If call directly use ones tensor
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

    output = prediction(torch.ones(1, 3, 224, 224), CovidPredictionModel)
    print('Output should be >> for Softmax:\n[0.07398344576358795, 0.62547367811203, 0.3005428612232208]\n')
    print('Output should be >> for LogSoftmax:\n[-2.6039137840270996, -0.4692460000514984, -1.202164888381958]\n')
    print('LogSoftmax output: (Smaller negative number has the highest probability)\n',output)