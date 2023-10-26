python train_meta.py --config configs/train_meta_fc.yaml --gpu 4,5,6,7



python train_classifier.py --config configs/train_classifier_FC100.yaml --gpu 0,1,2,3
python train_classifier.py --config configs/train_classifier_FC100_Resnet12.yaml --gpu 0,1,2,3

python train_classifier.py --config configs/train_classifier_FC100_vit.yaml --gpu 0,1,2,3
python train_classifier.py --config configs/train_classifier_FC100_vit200.yaml --gpu 4,5,6,7
python train_classifier.py --config configs/train_classifier_FC100_vit100.yaml --gpu 4,5,6,7

python train_classifier.py --config configs/train_classifier_FC100_Resnet152.yaml --gpu 0,1,2,3
python train_classifier.py --config configs/train_classifier_FC100_Resnet50.yaml --gpu 6,7