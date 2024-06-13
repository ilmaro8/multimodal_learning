# multimodal_learning

The scripts to train and use a multimodal learning architecture combining images (Whole Slide Images, WSIs) and texts (textual pathology reports). The architecture includes multiple input branches (a CNN backbone for images, a BERT backbone for reports) and is trained with a relatively limited amount of input samples (6000 couples images-reports), combining the classification of both modalities and self-supervised algorithms to align their representations. The architecture is tested on the classification of WSIs, outperforming the same architecture, trained only on the image-branch. Furthermore, the alignment among modality representations helps to exploit the architecture on a multimodal retrieval task (even if it s not explicitly trained on it) and on zero-shot learning.

## Requirements
Python==3.7.16, albumentations==0.1.8, numpy==1.19.3, pandas==1.3.5, pillow==9.4.0, torchvision==0.12.0, pytorch==1.11.0, transformers==4.6.1

## Reference
If you find this repository useful in your research, please cite:

[1] XXX.

Paper link: XXX

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

## model

## retrieval

## zeroshot learning
