# gp-retouch

`gp-retouch` is a Gaussian process-based image processing package.

This project is still in the very early stages of development. Its vision is to provide users with an easy-to-use toolbox for image processing, abstracting most of the low-level machine learning technicalities (à la scikit-learn). 

Your README draft is clear and well-structured! Here’s a revised version with minor grammatical improvements and stylistic tweaks:

gp-retouch is a Gaussian process-based image processing package.

This project is still in the very early stages of development. Its vision is to provide users with an easy-to-use toolbox for image processing, abstracting most of the low-level machine learning technicalities (à la scikit-learn).

The package also aims to evaluate the capabilities of Gaussian Processes (GPs) in the computer vision domain. We plan to create examples illustrating the impact of different kernels on model performance, hyperparameter initialization and learning, pre-training, the use of a parametric mean function, as well as the pros and cons of various sparse GP approaches.

Feel free to reach out if you have ideas! 😉

# TODOs

- [x] Lay out the basic project structure and classes.
- [x] Write a first reconstruction method.
- [ ] Migrate from PIL to Pillow.
- [ ] Make `loader.py` more robust.
- [ ] Implement additional pixel-dropping methods beyond random selection.
- [ ] For RGB images, explore correlations among channels during reconstructing.