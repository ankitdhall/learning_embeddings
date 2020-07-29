# [Hierarchical Image Classification using Entailment Cone Embeddings](https://ankitdhall.github.io/project/learning-representations-for-images-with-hierarchical-labels/)
<a href="https://ankitdhall.github.io" target="_blank">Ankit Dhall</a>, <a href="https://las.inf.ethz.ch/people/anastasia-makarova" target="_blank">Anastasia Makarova</a>, <a href="https://people.csail.mit.edu/oct/" target="_blank">Octavian Ganea</a>, <a href="http://da.inf.ethz.ch/people/DarioPavllo/" target="_blank">Dario Pavllo</a>, Michael Greeff, <a href="https://las.inf.ethz.ch/krausea" target="_blank">Andreas Krause</a>

![alt text](https://ankitdhall.github.io/publication/learning-representations-for-images-with-hierarchical-labels/featured_huc45c56e50f3be3419f4018ba4fe72357_373657_720x0_resize_lanczos_2.png "Jointly embeddings images and hierarchical labels on a Poincare disk in 2D")  
*Fig. 1: Jointly embeddings images and hierarchical labels on a Poincare disk in 2D*

More information about the project (paper, dataset and slides) can be found on the [project page](https://ankitdhall.github.io/project/learning-representations-for-images-with-hierarchical-labels/).

# Related publications and dataset
- [Learning Representations for Images With Hierarchical Labels, Master Thesis](https://arxiv.org/abs/2004.00909)
- [Hierarchical Image Classification using Entailment Cone Embeddings @CVPR 2020, DiffCVML workshop](https://arxiv.org/abs/2004.03459)
- [ETHEC Hierarchical dataset](https://www.researchcollection.ethz.ch/handle/20.500.11850/365379)

# Usage
Create a virtual environment using the `requirements.txt` file:
```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r learning_embeddings/requirements_3.6.txt
pip install opencv-python
pip install gitpython
```  

Use branch `Adam1x` to run experiments with the ETHEC dataset:  
`git checkout Adam1x`  

Splits for the ETHEC dataset can be found in `splits` folder. Experiments in this repository were conducted using **v0.1** of the ETHEC dataset.  

Sample command:  
`python learning_embeddings/network/ethec_experiments.py --experiment_name exp_test --experiment_dir exp --image_dir ETHEC_dataset_v0.1/ETHEC_dataset/IMAGO_build_test_resized/ --n_epochs 1 --model resnet18 --loss multi_level --set_mode train`

# Predicting Taxonomy for Organisms

![alt text](https://ankitdhall.github.io/project/learning-representations-for-images-with-hierarchical-labels/featured_hu84feb2bf561f49e98504fe25e8752a1b_2231317_720x0_resize_lanczos_2.png "The ETHEC dataset hierarchy")  
*Fig. 2: The ETHEC dataset hierarchy*

One of the main applications of this work is to assist natural history collections, museums and organizations that preserve large numbers of historical and extant biodiversity specimens to digitize and organize their collections. Hobbyists create their personal collections most of which are eventually donated to public institutions. Before integration, these specimens need to be sorted taxonomically by specialists who have little time and are expensive. If this resource intensive task could be preceded by a pre-sorting procedure, for instance, where these specimens are categorized by unskilled labour based on their family, sub-family, genus, and species it would expedite and economize the process.

Thanks to the the [ETH Library Lab](https://www.librarylab.ethz.ch/) the research conducted on the thesis will be turned into classification app that can be used by hobbyists, collectors, and researchers alike to speed up and economize classification and segregation of entomological specimens. More information about the app will be made available soon!

# References
If you find this useful for your research, please consider citing the following in your publication:
```
@misc{dhall2020hierarchical,
    title={Hierarchical Image Classification using Entailment Cone Embeddings},
    author={Ankit Dhall and Anastasia Makarova and Octavian Ganea and Dario Pavllo and Michael Greeff and Andreas Krause},
    year={2020},
    eprint={2004.03459},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{dhall2020hierarchical,
    title={Hierarchical Image Classification using Entailment Cone Embeddings},
    author={Ankit Dhall and Anastasia Makarova and Octavian Ganea and Dario Pavllo and Michael Greeff and Andreas Krause},
    year={2020},
    eprint={2004.03459},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@MISC{20.500.11850/365379,
	author = {Dhall, Ankit},
	publisher = {ETH Zurich},
	year = {2019},
	language = {en},
	copyright = {In Copyright - Non-Commercial Use Permitted},
	size = {5.98 GB},
	address = {Zurich},
	DOI = {10.3929/ethz-b-000365379},
	title = {ETH Entomological Collection (ETHEC) Dataset [Palearctic Macrolepidoptera, Spring 2019]},
}

```

![alt text](https://ankitdhall.github.io/project/learning-representations-for-images-with-hierarchical-labels/ec_2d_labels.png "Embedding label hierarchy with euclidean entailment cones in 2D")
*Fig. 3: Embedding label hierarchy with euclidean entailment cones in 2D*

