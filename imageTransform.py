import torch
import torchvision.transforms as transforms
from PIL import Image


def apply_transforms(img):
    """This function call is located in "makeInputforCNN" function."""

    # Normalization parameters for transforms.Normalize() transformation.
    mu = 0.570406436920166
    sigma = 0.1779220998287201

    # Library that used for convertion is important. 
    # Images that used to train model followed same convertion process which is:
    # img > PIL.convert('L') > PIL.convert('RGB')
    img = img.convert('L')
    img = img.convert('RGB')

    transformation = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((mu,), (sigma,))]
    )

    transformed_img = transformation(img)
    return transformed_img


def makeInputforCNN(img_path):
    """
    Argument:
        Test Image path
    Returns:
        Input tensor of a CNN Model
    """
    # open image
    img = Image.open(img_path)
    # apply transform
    transformed_img = apply_transforms(img)
    # check dimensions
    assert torch.is_tensor(transformed_img) == True, "Transformed image is not a torch tensor"
    # add dummy dimension to represent batch size for torch model
    transformed_img = transformed_img.unsqueeze(0)
    # check dimension
    assert transformed_img.shape == torch.Size([1, 3, 224, 224]), "model input image size is not valid"
    return transformed_img


if __name__ == '__main__':
    print("Image Transform module is running as main module for no reason.(Probably @.o)")
