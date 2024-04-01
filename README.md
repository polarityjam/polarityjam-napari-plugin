# polarityjam napari plugin

Start a napari instance for the feature extraction pipeline to study endothelial cell polarity, junctions and morphology.

Install Polarity-JaM from development banch. Install conda or micromamba.

Create an environment with `python=3.8` named `poalrityjam`

Activate the conda/micromamba environment:
```bash
conda activate polarityjam
```

Clone the Polarityjam feature extraction repository (https://github.com/polarityjam/polarityjam.git) and git checkout the dev branch.

Then install this plugin.

```bash
pip install git+https://github.com/polarityjam/polarityjam-napari-plugin.git
```

Start napari with the plugin:
```bash
polarityjam napari
```
