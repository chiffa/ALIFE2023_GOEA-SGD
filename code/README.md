# Complementary material and code for the ALIFE 2023 paper "Evolutionary Algorithms in the light of SGD: limit equivalence, minima flatness and fine-tuning"

## Generation of the data:

Please uncomment the relevant section in the `data_generation.py` file. Specifically, uncomment 
the desired section below the `if __name__ == "__main__"` line and set the experimental values 
in the `__init__ method` of the `Environment` class. Please note that GO-EA running on single 
machine searches in the population sequentially and has not been optimized. On the authors 
hardware it tested a single mutation in 3 seconds. For provided illustrations, tha (800 generations 
with a population of 20), that's over 14 hours of run. Intermediate savepoints are possible, but 
are controlled manually. 

Please note that only `CompressedNet` class implements all the auxilary methods required to 
perform the experiments mentioned in the paper. Setting other CovNets to `active_net` will 
result in a crash


## Figure rendering:
For provided data, the figures can be re-rendered by executing the `data generation.py` module, 
after correctly indicating the absolute path of the `run` directory containing the data after 
the `if __name__ == "__main__"` line.

Please note that the date indicated in the directories corresponds to the time when their 
content was frozen.

Similarly, note that it is possible to generate a grayscale-compatible version of figures by 
changing `active_cmap` from `Dark2` to `cividis`.


## Hardware used:

| Hardware | Spec |
| ----------- | ----------- |
| CPU | Intel i7-10700 @ 2.9 GHz |
| GPU | NVIDIA RTX 3080 |
| RAM | 32 G @ 2933 MT/s |

## Library versions:
Base Python version is 3.8.5, x64, provided by Anaconda.
All the dependencies are listed in `requirements.txt` with versions. 

In case problems are encountered with the figure generation, a separate `fig_gen_requirements.
txt` is provided, to be installed with Python 3.7.6, matching exactly the configuration of the 
machine on which the figures in the paper were generated. 

The environment can be tested with the provided `test.py` script.
