# python main.py \
#     --dataset CIFAR10 \
#     --recon_epochs 100 \
#     --epochs 100 \
#     --noise_ratio 0 \
#     --batch_size 64 \
#     --learning_rate 0.0001\
#     --device 8

python main.py \
    --dataset CIFAR10 \
    --recon_epochs 100 \
    --epochs 100 \
    --noise_ratio 0 \
    --batch_size 64 \
    --learning_rate 0.001\
    --optimizer adam\
    --device 0

python main.py \
    --dataset CIFAR10 \
    --recon_epochs 100 \
    --epochs 100 \
    --noise_ratio 0.25 \
    --batch_size 64 \
    --learning_rate 0.001\
    --optimizer adam\
    --device 0

python main.py \
    --dataset CIFAR10 \
    --recon_epochs 100 \
    --epochs 100 \
    --noise_ratio 0.5 \
    --batch_size 64 \
    --learning_rate 0.001\
    --optimizer adam\
    --device 0

python main.py \
    --dataset CIFAR10 \
    --recon_epochs 100 \
    --epochs 100 \
    --noise_ratio 0.75 \
    --batch_size 64 \
    --learning_rate 0.001\
    --optimizer adam\
    --device 0
