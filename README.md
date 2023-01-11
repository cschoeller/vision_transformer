## Vision Transformer for Tiny Imagenet

Implementation of the Vision Transformer from the paper:  
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", Dosovitskiy et al., 2021

To use this repository, first execute the `download_tinyimagenet.sh` and the `install_pycandle.sh` scripts before running `train.py`.

### Results

Currently the we achieve a top-1 accuracy of 39.9%, after self-supervised pretraining for 15 epochs on CIFAR10. In this pretraining we use our model to decode patch embeddings to reconstruct the original image. We randomly mask out patches of the input image that our model has to reconstruct in the decoding process. This incentivizes it to learn generalizing patterns and it regulariizes the pretraining. Without pretraining the model achieves about 38% accuracy.

Further improvements would be to test supervised pretraining that is supposed to be superior compared to the self-supervised regime. And additional data augmentation strategies such as MixUp and CutMix could be used.
