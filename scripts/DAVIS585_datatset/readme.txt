The DAVIS-585 dataset is available in our github page. For fair comparison, you should download our generate dataset to evaluate your model.
This repo contains the code for generating DAVIS-585 like test set with init masks.

Steps for generating DAVIS-585:
1. using select_images.py to sample images.
2. using generate_init_masks_480p.py to generate simulated initial masks.
* you should change the data path in corresponding files.