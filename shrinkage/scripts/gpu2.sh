# python main.py \
#     --dataset MNIST \
#     --recon_epochs 100 \
#     --epochs 100 \
#     --noise_ratio 0 \
#     --batch_size 64 \
#     --learning_rate 0.0001

# python main.py \
#     --dataset MNIST \
#     --recon_epochs 100 \
#     --epochs 100 \
#     --noise_ratio 0.25 \
#     --batch_size 64 \
#     --learning_rate 0.0001\
#     --optimizer adam

# python main.py \
#     --dataset MNIST \
#     --recon_epochs 100 \
#     --epochs 100 \
#     --noise_ratio 0.5 \
#     --batch_size 64 \
#     --learning_rate 0.0001\
#     --optimizer sgd

python main.py \
    --dataset MNIST \
    --recon_epochs 100 \
    --epochs 100 \
    --noise_ratio 0.75 \
    --batch_size 64 \
    --learning_rate 0.0001\
    --optimizer adam

# python main.py \
#     --dataset MNIST \
#     --recon_epochs 100 \
#     --epochs 100 \
#     --noise_ratio 1 \
#     --batch_size 64 \
#     --learning_rate 0.0001 \
#     --optimizer sgd
