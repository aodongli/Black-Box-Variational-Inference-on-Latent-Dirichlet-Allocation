# Black-Box-Variational-Inference-on-Latent-Dirichlet-Allocation
### BBVI on LDA

#### Features

* Data and model format follows from [David Blei's lda-c implementation](https://github.com/blei-lab/lda-c) exactly. Please refer to [David Blei's lda-c implementation](https://github.com/blei-lab/lda-c) for details.
* We reproduced the document log-likelihood formula in Python as lda-c for a fair comparison.
* We implement Adagrad extension.
* We provide detailed derivation for BBVI on LDA. Please see `report.pdf`.
* The results are evaluated on [20 newsgroups](http://qwone.com/~jason/20Newsgroups/).

#### Run inference

* We have already estimated $\alpha$ and $\beta$ using [David Blei's lda-c implementation](https://github.com/blei-lab/lda-c)  and saved them in `./model/final.*` The training data is presplitted in 20 newsgroups. The testset is top 100 documents in 20 newsgroups' original testset.
* Run the inference

```
python lda_inference.py
```

* In `./data/`, `test_inf-lda-lhood` is vanilla variational inference by [David Blei's lda-c implementation](https://github.com/blei-lab/lda-c). `bbvi_test_likelihood_adagrad` is BBVI on LDA with Adagrad algorithm. `bbvi_test_likelihood_all_sample` is BBVI on LDA with SGD algorithm.

#### Parallel inference extension

* We also extend it into parallel version `parallel_lda_inference.py`. 

* We have two options, multithreading and multiprocessing, specified by the whether or not using the argumetn `--thread`. **But multithreading does not work well in Python due to Global Interpreter Lock**.

* To specify the number of threads(processes) `--numthreads n`.

* For example, do multithreading with 5 threads 

  ```
  python parrallel_lda_inference.py --numthreads 5 --thread
  ```

  do multiprocessing with 5 processes

  ```
  python parrallel_lda_inference.py --numthreads 5
  ```

  Inference results are in `./data`.

* A speedup results on a sample of test dataset is shown below.

  ![paral_results](/Users/owner/Desktop/interested courses/Multicore System/project/docs/presentation/paral_results.jpg)

## Bugs report

Please email al5350@nyu.edu or stamdlee@outlook.com for any bug reports, unclear implementations, and suggestions.