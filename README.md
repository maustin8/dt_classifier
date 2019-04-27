# dt_classifier

This is a machine learning library developed by Max Austin for CS5350/6350 in University of Utah

Note: must be using a python version less than python3

# classifier_main
Note: classifier_main.py must be in the same directory as datasets/ and datasets/ must contain directories car/ and bank/ which must contain the train and test csv files for each dataset </br>

Note: ensemble types other than 'tree' take a substantial amount of time to run, especially with bank dataset

# To run program:  
Commands:  
Required:
"dataset": Dataset subfolder to choose {car or bank or tennis} </br>
Optional </br>
"-h", "--help": Show required and optional arguments for program </br> 
"-e", "--ensemble_type": Choose the type of ensemble learning {tree, adaboost, bagged, rand_forest} [DEFAULT = tree]</br>
"-t", "--num_iterations": Choose number of iterations [DEFAULT = 1000]</br> 
"-n", "--num_trees":Choose number of trees for bagged methods [DEFAULT = 1000]</br> 
"-s", "--sample_size": Choose sample size for bagged methods [DEFAULT = 2]</br> 
"-g", "--gain": Choose information gain variant to use [DEFAULT is entropy] {entropy, gini, majority_error} [DEFAULT = entropy]</br> 
"-d", "--tree_depth": Specify maximum tree depth where decision tree will stop building </br> 
"-f", "--feature_sample_size": Specify feature sample size for random forest </br>
"-u", "--unknown": Flag to replace unknown values with most common one </br> 
  </br>
example: </br> 
python classifier_main.py bank -g gini -d 16 -u


# linear_regression_main
Note: linear_regression_main.py must be in the same directory as datasets/ and datasets/ must contain directories concrete/ which must contain the train and test csv files 

# To run program:  
Commands:  
Optional </br>
"-d", "--descent": Choose method linear regression descent {batch, stochastic} [DEFAULT = batch] </br>
"-r", "--learn_rate": Choose the learning rate [DEFAULT = 1] </br>
"-n", "--num_iter": Choose the number of iterations for the gradient algorithms [DEFAULT = size of data</br>
</br>
example: </br>
python linear_regression_main.py -d stochastic -r 0.25 -n 2000

# percept_main
Note: percept_main.py must be in the same directory as datasets/ and datasets/ must contain directories bank-note/ which must contain the train and test csv files 

# To run program:  
Commands:  
Optional </br>

"-p", "--percept": Choose the type of perceptron to perform {standard, voted, average} [DEFAULT = standard] </br>
"-r", "--learn_rate": Choose the learning rate [DEFAULT = 1] </br>
"-e", "--num_epoch": Choose the number of epochs for training [DEFAULT = 10] </br>
</br>
example: </br>
python percept_main.py -p voted -r 0.05 -e 10

# svm_main
Note: nvm_main.py must be in the same directory as datasets/ and datasets/ must contain directories bank-note/ which must contain the train and test csv files 

# To run program:  
Commands:  
Optional </br>

"-s", "--svm": Choose the form of SVM to perform {primal, dual} [DEFAULT = primal] </br>
"-r", "--learn_rate": Choose the learning rate. Can be mutliple values that will be ran through the learning algorithm in sequence [DEFAULT] = [1] </br>
"-d", "--learn_rate_tweak": Choose the learning rate tweak \"d\" [DEFAULT] = 0 </br>
"-c", "--hyper_param": Choose the hyperparameter. Can be mutliple values that will be ran through the learning algorithm in sequence [DEFAULT] = default from assignment listing </br>
"-e", "--num_epoch": Choose the number of epochs for training [DEFAULT] = 100 </br>
"-k", "--kernel": Flag to implement Gaussian kernel in the dual learning optimization </br>

example: </br>
python svm_main.py -s dual -r 0.05 0.025 -e 100 -k

# nn_main
Note: nvm_main.py must be in the same directory as datasets/ and datasets/ must contain directories bank-note/ which must contain the train and test csv files 

# To run program:  
Commands:  
Optional </br>

"-r", "--learn_rate":Choose the initial learning rate [DEFAULT] = 1")
"-d", "--learn_rate_tweak":Choose the learning rate tweak \"d\" [DEFAULT] = 1")
"-w", "--width":Choose the width of each hidden layer [DEFAULT] = default from assignment listing")
"-e", "--num_epoch":Choose the number of epochs for training [DEFAULT] = 100")
"-l", "--depth": Choose the depths of the neural network for bonus [DEFAULT] = default from assignment listing")
"-tf", "--tensor": Flag to use tensorflow library for neural network implementation")

example: </br>
python n_main.py -w 5 10 25 50 50 -e 100