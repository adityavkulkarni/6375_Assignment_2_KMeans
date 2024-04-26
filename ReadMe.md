# CS6375 - Machine Learning
## Assignment-2 Part 2
___
### Table of Contents:
<!-- TOC -->
* [Table of Contents:](#table-of-contents)
* [Dataset:](#dataset-)
  * [File Used : gdnhealthcare.txt](#file-used--gdnhealthcaretxt)
* [Execution Instructions:](#execution-instructions)
* [File Structure:](#file-structure)
<!-- TOC -->
___
### Dataset: 
#### File Used : gdnhealthcare.txt
Other data files are stored in  data folder and can be used by passing file name to the script
___
### Execution Instructions:
- Execution:
  ```bash
  python main.py main.py --file file_path --max_k max_k
  ```
  ```
  optional arguments:
    --file path to the txt file (default: data/gdnhealthcare.txt)
    --max_k maximum number of clusters to use: results will be generated from k=3 to max_k (default: 10)
  ```
- All necessary packages are listed in [requirements.txt](requirements.txt)  

___
### File Structure:
- [ReadME.md](ReadME.md) 
- [data](data): directory for datasets 
- [main.py](main.py) : main file 
- [k_means_clustering.py](k_means_clustering.py) : file containing K means cluster class
- [Assignment2-CS6375.pdf](Assignment2-CS6375.pdf) : assignment description 
- [results](results) : directory for storing output 
  - [all](results/all): directory for graphs and results of all datasets
  - [k_means_elbow_gdnhealthcare.png](results/k_means_elbow_gdnhealthcare.png): plot of K vs SSE for data/k_means_elbow_gdnhealthcare.txt
  - [k_means_elbow_gdnhealthcare.csv](results/k_means_gdnhealthcare.csv): CSV contaning K, SSE and cluster distribution for data/k_means_elbow_gdnhealthcare.txt
- [CS6375_Assignment2_report](CS6375_Assignment2_report.pdf) : report for the assignment
