## Vision Transformer for Tiny Imagenet

Implementation of the Vision Transformer from the paper:  
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", Dosovitskiy et al., 2021

And the ConvNext from the paper:  
"A ConvNet for the 2020s", Liu et al., 2022.

To use this repository, first execute the `download_tinyimagenet.sh` and the `install_pycandle.sh` scripts before running `train.py`.

### Results

With the Vision Transformer, I currently achieve a top-1 accuracy of 40.67%, after self-supervised pretraining for 15 epochs on CIFAR10. In this pretraining I make the model decode patch embeddings to reconstruct the original image. I randomly mask out patches of the input image that the model has to reconstruct in the decoding process. This incentivizes it to learn generalizing patterns and it regularizes the pretraining. Without pretraining the model achieves about 38% accuracy.

The ConvNext reaches a top-1 accuracy of 48.80%, without any pretraining. It is better suited for the training on small datasets, most likely because conv nets have a stronger inductive bias for processing image data.

Further improvements for the performance of the ViT could be to test supervised pretraining that is supposed to be superior compared to the self-supervised regime. Additional data augmentation strategies such as MixUp and CutMix could also be used.
