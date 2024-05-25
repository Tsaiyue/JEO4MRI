#### Joint Edge Optimization Deep Unfolding Network for Accelerated MRI Reconstruction
##### ABSTRACT
Magnetic Resonance Imaging (MRI) is a widely used imaging technique, however it has the limitation of long scanning time. Though previous model-based and learning-based  MRI reconstruction methods have shown promising performance, most of them have not fully utilized the edge prior of MR images, and there is still much room for improvement. 
In this paper, we build a joint edge optimization model that not only incorporates individual regularizers specific to both the MR image and the edges, but also enforces a co-regularizer to effectively establish a stronger correlation between them. Specifically, the edge information is defined through a non-edge probability map to guide the image reconstruction during the optimization process. Meanwhile, the regularizers pertaining to images and edges are incorporated into a deep unfolding network to automatically learn their respective inherent a-priori information....

##### Enviroment prepare & pipeline building
```bash
# create the python / conda venv
conda create -n my_env_name python=3.8.6
python -m venv .venv

# activate the venv
source .venv/bin/activate
conda activate my_env

# install the framework with the soecific version 4 cuda & cudnn
conda install lightning -c conda-forge
python -m pip install lightning
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# install dependency
pip install -r requirements.txt

# try the training pipeline
python -m script/train.py
```

##### The accelerated reconstruction arch.
![arch.](https://github.com/Tsaiyue/JEO4MRI/assets/46399096/bd766dda-9e2d-4a68-9dde-a7ba85a70b82)

_Note_: The repo is WIP, Feel free to discuss with me by email tsaiyue01@gmail.com.
