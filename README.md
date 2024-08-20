# multimodal_learning

The scripts to train and use a multimodal learning architecture combining images (Whole Slide Images, WSIs) and texts (textual pathology reports). The architecture includes multiple input branches (a CNN backbone for images, a BERT backbone for reports) and is trained with a relatively limited amount of input samples (6000 couples images-reports), combining the classification of both modalities and self-supervised algorithms to align their representations. The architecture is tested on the classification of WSIs, outperforming the same architecture, trained only on the image-branch. Furthermore, the alignment among modality representations helps to exploit the architecture on a multimodal retrieval task (even if it s not explicitly trained on it) and on zero-shot learning.

## Requirements
Python==3.7.16, albumentations==0.1.8, numpy==1.19.3, pandas==1.3.5, pillow==9.4.0, torchvision==0.12.0, pytorch==1.11.0, transformers==4.6.1

## Reference
If you find this repository useful in your research, please cite:

[1] Marini N., Marchesin S. et al., Multimodal representations of biomedical knowledge from limited training whole slide images and reports using deep learning.

Paper link: https://www.sciencedirect.com/science/article/pii/S1361841524002287

## Pre-processing
### Images
WSIs are pre-processed following a similar approach to what proposed [here](https://github.com/ilmaro8/wsi_analysis).

### reports
Reports are stored as json files, with the following format:
{
    "ID_SAMPLES": {
        "diagnosis_nlp": "report content",
    },
    ...
}

## training and test
The multimodal architecture is trained to classify samples from input modalities (both images and texts) using the same classifier and to align the representation of coupled samples (train_classification.py). The architecture can also be trained only on the alignment of samples from different modalities (train_contrastive.py) implementing both the combination of self-supervised loss functions proposed in the paper and the CLIP algorithm.

## architecture
model.py and model_contrastive.py include the definition of the architecture. The architecture includes two input branches: a computer vision algorithm (many CNN MIL algorithms are implemented) to encode WSI representation and a branch to encode report representation. The single backbones can be changed according to user needs.

## retrieval
Multimodal retrieval performance of the model can be analyze with retrieval scripts. The retrieval can work across modalities, according to the input. It possible to retrieve similar images to an input text or viceversa. Multiple scripts help to evaluate different metrics.

## zeroshot learning
The multimodal architecture can be evaluated also on the zeroshot learning performance. Firstly, textual concepts feed the textual report branch (save_classes_zero_shot), to have an embedding represented a class. zero_shot_learning.py script evaluates similar patches (to link a concept to a small region) or WSIs. The only difference is the input embedding, since a WSI embedding has the same size as a patch embedding.
