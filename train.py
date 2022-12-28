import os

import PIL
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from vision_transformer import VisionTransformer
from resnet import ResNet
from simple_cnn import SimpleCNN


def save_model(model, checkpoint_path):
    model.eval()
    model_state_dict = model.state_dict()
    torch.save({'model_state_dict' : model_state_dict,
                }, checkpoint_path)

def count_correct_preds(y_pred, y_true):
    pred_indices = y_pred.max(1, keepdim=True)[1]    
    count_correct = pred_indices.eq(y_true.view_as(pred_indices)).sum().double()
    return count_correct

def evaluate(model, val_data):
    data_loader = data.DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)
    cnt_correct = 0
    for batch in data_loader:
        x, y = batch[0].cuda(), batch[1].cuda()
        y_pred = model(x)
        cnt_correct += count_correct_preds(y_pred, y)
    return cnt_correct/len(val_data)

class BicubicUpsampling():
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        return img.resize((self.size, self.size), PIL.Image.BICUBIC)

def load_dataset(path):
    # normalization taken from https://discuss.pytorch.org/t/data-preprocessing-for-tiny-imagenet/27793
    upsample = BicubicUpsampling(128)
    #crop_size = 100
    #center_crop = transforms.CenterCrop(crop_size)
    #rand_crop = transforms.RandomCrop(size=crop_size)
    #hflip = transforms.RandomHorizontalFlip(p=0.5)
    to_tensor = transforms.ToTensor()
    #normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #trfs_train = transforms.Compose([upsample, rand_crop, hflip, to_tensor, normalize])
    #trfs_val = transforms.Compose([upsample, center_crop, to_tensor, normalize])
    trfs_train = transforms.Compose([upsample, to_tensor])
    trfs_val = transforms.Compose([upsample, to_tensor])
    train = ImageFolder(os.path.join(path, 'train'), transform=trfs_train)
    val = ImageFolder(os.path.join(path, 'val'), transform=trfs_val)
    return train, val

def train_model(model, train_data):
    epochs = 50
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    cross_entropy_loss = nn.CrossEntropyLoss()
    data_loader = data.DataLoader(train_data, batch_size=96, shuffle=True, num_workers=12)
    for epoch in range(epochs):
        running_loss = 0
        for i, batch in enumerate(data_loader):
            x, y = batch[0].cuda(), batch[1].cuda()
            y_pred = model(x)
            error = cross_entropy_loss(y_pred, y)
            # training step
            running_loss += error.item()
            optimizer.zero_grad() # reset gradient
            error.backward() # compute new gradient
            optimizer.step() # grad descent step
            print(f"epoch {epoch+1}, batch {i+1}/{len(data_loader)}, loss {running_loss/(i+1)}")

def prepare_val_folder(dataset_path):
    """
    Split validation images into separate class-specific sub folders. Like this the
    validation dataset can be loaded as an ImageFolder.
    """
    val_dir = os.path.join(dataset_path, 'val')
    img_dir = os.path.join(val_dir, 'images')

    # read csv file that associates each image with a class
    annotations_file = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = annotations_file.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    annotations_file.close()

    # create class folder if not present and move image into it
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(os.path.dirname(img_dir), folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

    # remove old image path
    if os.path.exists(img_dir):
        os.rmdir(img_dir)

def main():
    dataset_path = "./tiny-imagenet-200"
    prepare_val_folder(dataset_path)
    model = VisionTransformer(img_size=128, patch_size=16, num_classes=200)
    #model = SimpleCNN()
    # model = ResNet()
    #model = models.resnet18(num_classes=200)
    #model = models.resnet18(pretrained=True)
    #model.fc = nn.Linear(model.fc.in_features, 200)
    model.cuda()
    train, val = load_dataset(dataset_path)
    train_model(model, train)
    save_model(model, 'model.pt')
    accuracy = evaluate(model, val)
    print(f"Classification accuracy {accuracy}")

if __name__ == "__main__":
    main()
