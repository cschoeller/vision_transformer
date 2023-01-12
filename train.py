import os
from dataclasses import dataclass
import random

import click
import PIL
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from pycandle.training.model_trainer import ModelTrainer
from pycandle.general.experiment import Experiment
from pycandle.training.callbacks import HistoryRecorder, ModelCheckpoint
from pycandle.general.utils import load_model

from vision_transformer import VisionTransformer
from convnext import ConvNext
from autoencoding_vision_transformer import AutoencodingVit


@dataclass
class Config:
    # pre-training
    pre_lr = 0.0003
    pre_epochs = 15
    pre_batch_size = 64
    pre_masking_p = 0.2
    pre_wdecay = 0.05

    # training
    train_lr = 0.0003
    train_epochs = 300
    train_batch_size = 128
    wdecay = 0.05

    # augmentations
    img_size = 64
    pre_crop_size = 80

_CONFIG = Config()


def count_correct_preds(y_pred, y_true):
    pred_indices = y_pred.max(1, keepdim=True)[1]    
    count_correct = pred_indices.eq(y_true.view_as(pred_indices)).sum().double()
    return count_correct

def evaluate(model, val_data):
    model.eval()
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

def get_train_transforms():
    # normalization from https://discuss.pytorch.org/t/data-preprocessing-for-tiny-imagenet/27793
    # define augmentations
    upsample_train = BicubicUpsampling(_CONFIG.pre_crop_size)
    upsample_val = BicubicUpsampling(_CONFIG.img_size)
    rand_aug = transforms.RandAugment(num_ops=5, magnitude=10)
    rand_rot = transforms.RandomRotation(degrees=(-25, 25))
    rand_crop = transforms.RandomCrop(size=_CONFIG.img_size)
    hflip = transforms.RandomHorizontalFlip(p=0.5)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    rand_erase = transforms.RandomErasing(p=0.3, scale=(0.02, 0.25))
    trfs_train = transforms.Compose([upsample_train, rand_rot, rand_crop, hflip, rand_aug, to_tensor, normalize, rand_erase])
    trfs_val = transforms.Compose([upsample_val, to_tensor, normalize])
    return trfs_train, trfs_val

def load_train_dataset(path):
    trfs_train, trfs_val = get_train_transforms()
    # load images with transformation function
    train = ImageFolder(os.path.join(path, 'train'), transform=trfs_train)
    val = ImageFolder(os.path.join(path, 'val'), transform=trfs_val)
    return train, val

def load_pretrain_dataset():
    print("Loading CIFAR10:")
    upsample_train = BicubicUpsampling(_CONFIG.pre_crop_size)
    upsample_val = BicubicUpsampling(_CONFIG.img_size)
    rand_aug = transforms.RandAugment(num_ops=5, magnitude=10)
    rand_rot = transforms.RandomRotation(degrees=(-25, 25))
    rand_crop = transforms.RandomCrop(size=_CONFIG.img_size)
    hflip = transforms.RandomHorizontalFlip(p=0.5)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    trfs_train = transforms.Compose([upsample_train, rand_rot, rand_crop, hflip, rand_aug, to_tensor, normalize])
    trfs_val = transforms.Compose([upsample_val, to_tensor, normalize])
    train = CIFAR10(".", train=True, transform=trfs_train, download=True)
    val = CIFAR10(".", train=False, transform=trfs_val, download=True)
    if os.path.exists("./cifar-10-python.tar.gz"):
        os.remove("./cifar-10-python.tar.gz")
    return train, val

def plot_history(history_recorder, output_path):
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
        plt.savefig(os.path.join(output_path, '{}.png'.format(key)))
        plt.close()

def prepare_val_folder(dataset_path):
    """
    Split validation images into separate class-specific sub folders. Like this the
    TinyImagenet validation set can be loaded as a torchvision ImageFolder.
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

def reconstruction_loss(batch, model):
    x, _ = batch
    bs, num_patches = x.shape[0], model.num_patches
    patch_mask = (torch.rand(size=(bs, num_patches)) < _CONFIG.pre_masking_p)
    y_pred = model(x, patch_mask)
    error = ((x - y_pred)**2).mean() # MSE
    return error, y_pred

def plot_reconstruction_examples(model, val, experiment):
    experiment.add_directory('reconstructions')
    mean = torch.Tensor([0.4914, 0.4822, 0.4465]).view(1,1,-1).cuda()
    std = torch.Tensor([0.247, 0.243, 0.261]).view(1,1,-1).cuda()
    model.cuda()
    with torch.no_grad():
        for i in random.sample(list(range(len(val))), 30):
            x, _= val[i]
            x = x.cuda()
            y_pred = model(x.unsqueeze(0)).squeeze()
            x = (x.permute(1,2,0) * std) + mean
            y_pred = (y_pred.permute(1,2,0) * std) + mean

            fig, ax = plt.subplots(1,2)
            fig.set_figheight(4)
            fig.set_figwidth(8)
            ax[0].imshow(x.cpu())
            ax[1].imshow(y_pred.cpu())
            plt.tight_layout()
            plt.savefig(os.path.join(experiment.reconstructions, f'{i}_example.png'))
            plt.close()

def pretrain_vit(model, experiment):
    """
    Pretrain the vit self-supervised as a patch-wise autoencoder. Apply a input patch masking to incentivize the
    model to learn generalizing patterns, enable information flow between patch encodings in the
    attention layers, and regularize it.
    NOTE: According to the vit authors supervised pre-training (classification) works better. But self-supervised
    pre-training is more universally applicable and hence was more interesting to explore.
    """
    print("Run pretraining")
    model.enable_pretrain(True)
    train, val = load_pretrain_dataset()

    # define training setting
    optimizer = optim.AdamW(model.parameters(), lr=_CONFIG.pre_lr, weight_decay=_CONFIG.pre_wdecay)
    train_loader = data.DataLoader(train, batch_size=_CONFIG.pre_batch_size, shuffle=True, num_workers=12)
    val_loader = data.DataLoader(val, batch_size=_CONFIG.pre_batch_size, shuffle=True, num_workers=12)

    # setup and run trainer
    trainer = ModelTrainer(model, optimizer, reconstruction_loss, _CONFIG.pre_epochs, train_loader, val_loader,
                           custom_model_eval=True, device=0, scheduler=None)
    history_recorder = HistoryRecorder()
    trainer.add_callback(history_recorder)
    model_checkpoint_name = 'pretrain_checkpoint.pt'
    trainer.add_callback(ModelCheckpoint(experiment.models, model_name=model_checkpoint_name))
    trainer.start_training()
    experiment.add_directory('pretrain_plots')
    plot_history(history_recorder, experiment.pretrain_plots)

    # load best performing checkkpoint
    load_model(model, experiment.models + "/" +model_checkpoint_name)
    plot_reconstruction_examples(model, val, experiment)
    model.enable_pretrain(False)

def train_vit(model, experiment, train, val):
    """ Train the vit to classify TinyImagenet. """
    # define training setting
    optimizer = optim.AdamW(model.parameters(), lr=_CONFIG.train_lr, weight_decay=_CONFIG.wdecay)
    loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    train_loader = data.DataLoader(train, batch_size=_CONFIG.train_batch_size, shuffle=True, num_workers=12)
    val_loader = data.DataLoader(val, batch_size=_CONFIG.train_batch_size, shuffle=True, num_workers=12)

    # setup and run trainer
    trainer = ModelTrainer(model, optimizer, loss, _CONFIG.train_epochs, train_loader,
                           val_loader, device=0, scheduler=None)
    history_recorder = HistoryRecorder()
    trainer.add_callback(history_recorder)
    trainer.add_metric(accuracy_metric)
    trainer.add_callback(ModelCheckpoint(experiment.models, target_metric='val_accuracy_metric', smallest=False))
    trainer.start_training()
    plot_history(history_recorder, experiment.plots)

    # load best performing checkkpoint
    load_model(model, experiment.models + '/model_checkpoint.pt')

@click.command()
@click.option('--name', default='test')
@click.option('--load-pretrained', 'pretrained_path')
@click.option('--pretrain/--no-pretrain', default=True)
@click.option('--use-convnext/--no--convnext', default=False)
def main(name, pretrained_path, pretrain, use_convnext):
    # create experiment
    experiment = Experiment(experiment_name=name)
    experiment.add_directory('models')

    # model
    if use_convnext:
        model = ConvNext(img_size=_CONFIG.img_size, num_classes=200, blocks=(3,6,3),
                         channels=(96, 192, 384))
        pretrain = False
        _CONFIG.wdecay = 0.
    else:
        model = AutoencodingVit(img_size=_CONFIG.img_size, patch_size=8, num_classes=200,
                                 dim=512, depth=12, heads=12, mlp_dim=512)
    model.cuda()
    model.train()

    # self-supervised pre-training on CIFAR10
    if pretrained_path:
        load_model(model, pretrained_path)
    elif pretrain:
        pretrain_vit(model, experiment)

    # load target dataset
    dataset_path = "./tiny-imagenet-200"
    prepare_val_folder(dataset_path)
    train, val = load_train_dataset(dataset_path)
    train_vit(model, experiment, train, val)

    # evaluate model
    accuracy = evaluate(model, val)
    print(f"Best validation accuracy: {accuracy}")

if __name__ == "__main__":
    main()
