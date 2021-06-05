#To overcome over-fitting, we fine-tuned the hyperparameter settings in the few-shot regime.
#To overcome over-smoothing, we fine-tuned the hyperparameter settings when the coarsening ratio is 0.1.
#example, the coarsening ratio is 0.5
python train.py --dataset cora --experiment fixed --coarsening_ratio 0.5
python train.py --dataset cora --experiment few --epoch 100 --coarsening_ratio 0.5
python train.py --dataset citeseer --experiment fixed --coarsening_ratio 0.5
python train.py --dataset pubmed --experiment fixed --coarsening_ratio 0.5
python train.py --dataset pubmed --experiment few --epoch 100 --coarsening_ratio 0.5
python train.py --dataset dblp --experiment random --epoch 200 --early_stopping 0 --K 20 --alpha 0.05 --coarsening_ratio 0.5
python train.py --dataset Physics --experiment random --epoch 500 --lr 0.0005 --weight_decay 0 --K 20 --alpha 0.1 --coarsening_ratio 0.5
python train.py --dataset Physics --experiment few --epoch 500 --lr 0.001 --weight_decay 0 --K 20 --alpha 0.1 --coarsening_ratio 0.5


