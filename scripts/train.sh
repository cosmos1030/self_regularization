# python train.py \
#     --gpu 5 \
#     --dataset cifar10 \
#     --batch-size 64 \
#     --lr 0.0001 \
#     --optimizer sgd \
#     --epochs 100 \
#     --train-percent 1 \
#     --label-noise 0 \
#     --noise-seed 777 \
#     --recon-epochs 25 50 75 100 \
#     --plot-spectral \
#     --jse \
#     --seed 200

python train.py \
    --gpu 9 \
    --dataset cifar10 \
    --batch-size 64 \
    --lr 0.0001 \
    --optimizer sgd \
    --epochs 100 \
    --train-percent 1 \
    --label-noise 0 \
    --noise-seed 777 \
    --recon-epochs 1 25 50 75 100 \
    --plot-spectrum \
    --jse \
    --seed 200