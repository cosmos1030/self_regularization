# Eigenvector-Self-Regularization

mlp3.py trains a 3 layer multilayer perceptron on CIFAR-10 dataset, and collects the leading eigenvectors of the three layers after each epoch, the spectrum of the weight matrices of each layer after each epoch and the accuracy using the test set after each epoch.

leading_eigenvectors_best_fit_geodesic.py gives the RSS and fitting score of the best fit geodesic, and also outputs the plot of the geodesic parameters of the projected data on the geodesic against the epochs.

leading_eigenvectors_dim_reduction.py gives the RSS and fitting score of the best fit two sphere, and also visualize the projected points on a 2 sphere.

