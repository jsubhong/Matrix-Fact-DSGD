# Matrix-Fact-DSGD
Matrix Factorisation using Distributed Stochastic Gradient Descent - Implemented Using PySpark 1.3.0
- uses PySpark to perform distributed stochastic gradient descent on training matrix V in csv format:
x1,y1,data1
x2,y2,data2
x3,y3,data3 etc...

example usage:
spark-submit dsgd_mf.py 20 3 100 0.9 0.1 autolab_train.csv w.csv h.csv

parameters:
1. number of factors - how many latent factors desired in factored matrices W, and H 
2. number of workers - desired workers for parallelization
3. number of iterations - number of updates desired 
4. BETA - gradient descent step size
5. LAMBDA - L2 regularization term, cross-validate to tune
6. input file V matrix - training data input for matrix V as csv
7. output W matrix path - factored TALL W matrix
8. output H matrix path - factored FAT H matrix
