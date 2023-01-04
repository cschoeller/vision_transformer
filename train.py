import os

import PIL
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
#import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from pycandle.training.model_trainer import ModelTrainer
from pycandle.general.experiment import Experiment
from pycandle.training.callbacks import HistoryRecorder, ModelCheckpoint

from vision_transformer import VisionTransformer


def load_model(model, checkpoint_path):
    loaded_state_dict = torch.load(checkpoint_path)['model_state_dict']
    model.load_state_dict(loaded_state_dict)

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

def accuracy_metric(y_pred, y_true):
    return float(count_correct_preds(y_pred, y_true)/len(y_true))

class BicubicUpsampling():
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        return img.resize((self.size, self.size), PIL.Image.Resampling.BICUBIC)

def load_dataset(path):
    # normalization from https://discuss.pytorch.org/t/data-preprocessing-for-tiny-imagenet/27793
    # define augmentations
    img_size = 64
    upsample_train = BicubicUpsampling(80)
    upsample_val = BicubicUpsampling(img_size)
    rand_aug = transforms.RandAugment(magnitude=9)
    rand_rot = transforms.RandomRotation(degrees=(-25, 25))
    rand_crop = transforms.RandomCrop(size=img_size)
    hflip = transforms.RandomHorizontalFlip(p=0.5)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    rand_erase = transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
    # compose transformation
    trfs_train = transforms.Compose([upsample_train, rand_rot, rand_crop, hflip, rand_aug, to_tensor, normalize, rand_erase])
    trfs_val = transforms.Compose([upsample_val, to_tensor, normalize])
    # load images with transformation function
    train = ImageFolder(os.path.join(path, 'train'), transform=trfs_train)
    val = ImageFolder(os.path.join(path, 'val'), transform=trfs_val)
    return train, val

def plot_history(history_recorder, experiment):#, testset_name=None):
    for key in history_recorder.history.keys():
        if 'lr' in key:
            continue
        if 'val' in key:
            continue
        plt.figure()
        plt.plot(history_recorder.history[key])
        val_key = 'val_' + key
        if val_key in history_recorder.history:
            plt.plot(history_recorder.history['val_' + key])
        plt.title('model ' + key)
        plt.ylabel(key)
        plt.xlabel('Epoch')
        if val_key in history_recorder.history:
            plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig(os.path.join(experiment.plots, '{}.png'.format(key)))
        plt.close()

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
    experiment = Experiment(experiment_name='vit')
    experiment.add_directory('models')

    dataset_path = "./tiny-imagenet-200"
    prepare_val_folder(dataset_path)
    train, val = load_dataset(dataset_path)

    model = VisionTransformer(img_size=64, patch_size=8, num_classes=200, dim=512, depth=12, heads=12, mlp_dim=512)
    model.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    epochs = 100
    train_loader = data.DataLoader(train, batch_size=128, shuffle=True, num_workers=12)

    val_loader = data.DataLoader(val, batch_size=128, shuffle=True, num_workers=12)
    trainer = ModelTrainer(model, optimizer, loss, epochs, train_loader, val_loader, device=0, scheduler=None)
    history_recorder = HistoryRecorder()
    trainer.add_callback(history_recorder)
    trainer.add_callback(ModelCheckpoint(experiment.models))
    trainer.add_metric(accuracy_metric)
    trainer.start_training()
    plot_history(history_recorder, experiment)

    checkpoint_path = experiment.models + '/model_checkpoint.pt'
    load_model(model, checkpoint_path)
    model.eval()
    accuracy = evaluate(model, val)
    print(f"Best validation model accuracy: {accuracy}")

if __name__ == "__main__":
    main()
