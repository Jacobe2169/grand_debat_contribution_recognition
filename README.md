# Contribution recognition in the Grand Débat National 


## Setup

**Attention** Since, the crf library used is based on an older version of `scikit-learn`
it is advised to use virtual environements using `conda` or `virtualenv`.

Setup dependencies using the `requirements.txt` file

```shell
pip install -r requirements.txt
```

## Train the model

First, get the export CSV file from the `label-studio` projet.

Then, simply run the `train_crf.py` using the following command:

```shell
python train_crf.py <export_filename> <output_filename>
# python train_crf project-4-at-2022-08-29-06-39-2333008e.csv crf_model.pkl
```

## Use the model on the Grand Debat Data

First, download the export (csv format) from the [official website](https://granddebat.fr/pages/donnees-ouvertes).
There is an export for each theme (taxes, democracy, government organisation, ecological transition).

Second, use the `predict_crf.py`

```shell
python predict_crf.py <gdb_csv_export_filename> <crf_model_filename> <output_directory_path>
#python predict_crf.py data/LA_TRANSITION_ECOLOGIQUE.csv crf_model.pkl test_output_dir
```


## Author

Jacques Fize