# A Confidence Machine for Sparse High-order Interaction Model


This package implements both computationally and statistically efficient solutions to compute 
the “exact full conformal prediction set (full-CP)” of a Sparse High-order Interaction Model (SHIM).
To the best of our knowledge, this is the first attempt to compute an exact full-CP of a 
complex model such as a SHIM. This is also the first attempt to construct a conformal prediction
set in the context of a pattern mining model which is fitted with a branch and bound approach. 
As of today, an efficient exact full-CP is possible only for simple regression models such as LASSO, ridge regression, and ordinary least square regression. The proposed method adds SHIM to that list.

See our paper https://arxiv.org/pdf/2205.14317.pdf for more details.

## Installation & Requirements

This package has the following requirements:

- [numpy](http://numpy.org)
- [scikit-learn](http://scikit-learn.org)
- [pandas](https://pandas.pydata.org)
- [python-intervals](https://pypi.org/project/python-intervals/)
- [matplotlib](https://matplotlib.org/)

We recommend installing or updating anaconda to the latest version and use Python 3 (We used Python 3.9.7).

All commands are run from the terminal.

## Reproducibility

**NOTE**: Due to the randomness of data generating process, we note that the results might be slightly different from the paper. However, the overall results for interpretation will not change.

All the figure results are saved in folder "/results"


To reproduce the results using synthetic data please run the following scripts:

- Table1 and Table2
	```
	>> python ex1_synthetic.py
	```

- Figure 2
	```
	>> python ex2_synthetic.py
	```

- Figure 3 (Left)
	```
	>> python ex3_synthetic.py
	```
	
- Figure 3 (Right)
	```
	>> python ex4_synthetic.py
	```
	
- Figure 4 (3tc)
	```
	>> python ex1_3tc.py
	
- Figure 4 (abc)
	```
	>> python ex1_abc.py
	
	
- Figure 5 (friedman2)
	```
	>> python ex1_friedman2_continuous.py
	
- Figure 5 (bodyfat)
	```
	>> python ex1_bodyfat.py




