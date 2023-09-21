# Brain tumor classification using CNN

## Dataset

The dataset was taken from [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Dataset details

This dataset contains 7023 images of human brain MRI images which are classified into 4 classes: glioma - meningioma - no tumor and pituitary.

This dataset is a combination of the following three datasets :
* [figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
* [SARTAJ dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
* [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection?select=no)

## Model

I trained 2 differents models:

* using tensorflow: it has achieved 98.25% accuracy on test set.

* using pytorch: it has achieved 98.3% accuracy on test set.

Both models were created via transfer learning, ResNet50 architecture with pretrained weights from ImageNet

Pretrained weights you can find here: [tensorflow](https://drive.google.com/file/d/1Zeu98VqxFIbdszcNGY6x37Kf-nKwH0nD/view?usp=sharing), [pytorch](https://drive.google.com/file/d/1CZRqq7DtojEZ67ZTkJf0qOJy5DOe2AEV/view?usp=sharing)