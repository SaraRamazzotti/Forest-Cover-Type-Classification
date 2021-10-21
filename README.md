## Forest cover type classification using AdaBoost
Download the [Cover Type Dataset](https://www.kaggle.com/uciml/forest-cover-type-dataset) for multiclass classification. Implement AdaBoost from scratch and run it using decision stumps (binary classification rules based on single features) as base classifiers to train seven binary classifiers, one for each of the seven classes (one-vs-all encoding). Use external cross-validation to evaluate the multiclass classification performance (zero-one loss) for different values of the number T of AdaBoost rounds. In order to use the seven binary classifiers to make multiclass predictions, use the fact that binary classifiers trained by AdaBoost have the form ![formula](https://render.githubusercontent.com/render/math?math=h(x)=sgn\(g(x)\)) and predict using ![formula](https://render.githubusercontent.com/render/math?math=argmax_ig_i(x)) where ![formula](https://render.githubusercontent.com/render/math?math=g_i) corresponds to the binary classifier for class *i*.

### Jupyter Notebook
The notebook of the analysis was runned on google colab but if you want to run it in local you have to enter in the project's directory and create the virtual environment. Then, you can enter the virtual environment and install all the needed dependencies.
```
$ pipenv install
$ pipenv shell
$ pip install .
```

### Report
In the report folder you can find a report with some considerations on the experiments.