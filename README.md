## Vision Transformer for Tiny Imagenet

Implementation of the Vision Transformer from the paper:  
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", Dosovitskiy et al., 2021

To use this repository, first execute the `download_tinyimagenet.sh` and the `install_pycandle.sh` scripts before running `train.py`.

Currently we achieve an top-1 accuracy of about 37%. However, in the checkpoint I save the model with he lowest validation loss.
Curiously, this is not necessarily the model with the best validation accuracy (~33%). Probably this happens because at later training stages the
model becomes more confident about certain samples that lead to better thresholded accuracy, but worse overall validation loss (at the tail end, after the [double descent](https://openai.com/blog/deep-double-descent/)).

The next improvement I want to try is pre-training the model self-supervised on a different dataset. Another way to improve it further
could be the use of more advanced augmentations, like MixUp or CutMix.
