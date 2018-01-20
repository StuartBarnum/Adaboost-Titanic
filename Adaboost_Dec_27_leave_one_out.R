#Running script, assuming that the Kaggle training and test datasets are in the working directory 
#and named "Kaggle_train.csv" and "Kaggle_test.csv", and that 3 cores are available for parallel 
#processing (the program implements leave-one-out cross-validation in parallel). The script consists 
#of a definition of a function that handles the data wrangling, a definition of a function that 
#implements adaboost (adaptive boosting) over a CART algorithm (implemented with the rpart package),
#and the main program, which implements the leave-one-out cross validation. (For each of the 
#examples in the Kaggle training dataset, the Adaboost algorithm is implemented over the set of all 
#of the other examples in the Kaggle training dataset, with the models thus derived used to make a 
#prediction over the one example that has been "left out.") The printed output of the program 
#consists of both confusion matrices and average accuracies based on the these predictions, accross 
#the entire Kaggle training dataset (with one confusion matrix and one average for each of the 
#numbers of iterations of the adaboost algorithm). The output also includes a list, model_info_list, 
#that contains one element for each iteration of the cross validation, and allows examination of the 
#rpart models and the adaboost weights and adjustment factors. Depending on the settings one may 
#choose at the top of the main program (together with an element of chance), maximum accuracy 
#(between 0.81 and 0.84) on the examples "left out" appears to occur at between 3 and 20 adaboost 
#iterations. Also depending on the settings (particularly the number the number of adaboost 
#iterations), the size of model_output_list can be as great as several gigabytes.

library(tidyverse)

#For import of the downloaded Kaggle training and test sets:
#setwd("~/Desktop/R files/Adaboost-Titanic")

#For reading in the data and testing elements of the function titanic_wrangle:
#train <- read_csv("Kaggle_train .csv")
#test <- read_csv("Kaggle_test.csv")

#The function titanic_wrangle is designed to work both with the Kaggle training and 
#test sets and with any partition of the Kaggle training set into "training" and 
#"test" sets, for the purpose of cross validation.
titanic_wrangle <- function(train, test) {

#Each of train and test is a data frame, perhaps with the output values 
#in the Test data frame removed (to remove bias due to added columns
#with calculated values that may directly reflect the output data). 
  
  library(stringr, stringi)
  
  #convert the "Survived" values in the test data frame to NA
  #(or leave them as such if they are already NA), after saving these
  #values in a vector (to be restored to the test data frame  after the 
  #wrangling is finished). This step is intended to help avoid bias in cross
  #validation, due to calculation of new columns on the basis of the 
  #correct output (the frac_Surname_Survived variable that will be introduced
  #in the wrangling)
  if ("Survived" %in% names(test)) {
    test_Survived <- test$Survived
  } else {
    test_Survived <- NA
  }
  test$Survived <- NA
  
  #For purposes of creating calculated values, combine the training and test dataframes
  combined_dataset <- rbind(train, test)
  combined_dataset <- as.tibble(combined_dataset)
  
  #append the column for the sum of the numbers of siblings and parents
  combined_dataset_plus <- combined_dataset %>%
    mutate(SibSpPlusParch = SibSp + Parch)
  
  #extract the titles from the passenger names ("Mr.", etc.) and append a new column
  #with the title, conditional on there being at least one person with the title in
  #the test set.
  names <- combined_dataset_plus$Name
  commas <- (str_locate(names, ","))[,"start"]
  periods <- (str_locate(names, "\\."))[,"start"]
  Title <- str_sub(names, commas + 2, periods)
  combined_dataset_plusplus <- combined_dataset_plus %>%
    mutate(Title = as.factor(Title))
  rm(Title, periods, commas, names)
  
  #change to NA any value of Title which is equal to "Dona." or "Ms." (cases with only
  #one or two instances in the dataset). "Col.", with only four instances (two in the 
  #training set) is also rather sketchy, but I am leaving it for now.
  combined_dataset_plusplus <-
    combined_dataset_plusplus %>%
    mutate(Title = replace(Title,
                           Title == "Dona." | Title == "Ms.",
                           NA))
  
  #extract some relevant information from listed cabins
  #(ignoring the matter of whether more than one cabin is listed)
  cabins <- combined_dataset_plusplus$Cabin
  Cabin_letter <- str_sub(cabins, 1, 1)
  combined_dataset_plusplusplus <- combined_dataset_plusplus %>%
    mutate(Cabin_letter)
  rm(cabins, Cabin_letter)

  #replace NAs in the Age column with averages within the classes defined 
  #by Title, Pclass, and Survived
  combined_dataset_plusplusplus <- combined_dataset_plusplusplus %>%
    group_by(Title, Pclass, Survived) %>%
    mutate(Age = ifelse(is.na(Age), mean(Age, na.rm = TRUE), Age)) %>%
    ungroup()
  
  #combined_dataset_plusplusplus <- combined_dataset_plusplusplus %>% 
  #  replace_na(list(Cabin_letter = "U"))
  
  combined_dataset_plusplusplus$Cabin_letter <- 
    as.factor(combined_dataset_plusplusplus$Cabin_letter)
  
  combined_dataset_plusplusplus$Embarked <- 
    as.factor(combined_dataset_plusplusplus$Embarked)
  
  combined_dataset_plusplusplus$Pclass <- 
    as.factor(combined_dataset_plusplusplus$Pclass)
  
  combined_dataset_plusplusplus$Sex <- 
    as.factor(combined_dataset_plusplusplus$Sex)
  
  combined_dataset_plusplusplus$Survived <- 
    as.factor(combined_dataset_plusplusplus$Survived)
  
  #extract the surnames (for calculations in the data to follow)
  library(stringi)
  full_name <- combined_dataset_plusplusplus$Name
  comma_pos <- stri_locate_first(full_name, regex = ",")[,"start"]
  Surname <- str_sub(full_name, 1, comma_pos - 1)
  combined_dataset_plusplusplus <- combined_dataset_plusplusplus %>%
    mutate(Surname = as.factor(Surname))
  rm(full_name, comma_pos, Surname)
  
  #The "population" of the Titanic is much less than, say, the population of a country,
  #and families may tend to travel togother. Therefore, on the hypthesis that there may be 
  #a relevant relationship, for each invividual, between survival of that individual and 
  #survival of other individuals with the same surname (because the latter's association 
  #with membership in the same family), we now, for each individual, count the number of
  #people with the same Surname as that individual, and the fraction, among individuals 
  #reflected in the training set (Survived != NA in the present table), who survived.
  #However, (reiterating) this presents a circularity or overfitting issue that must be
  #handled carefully.
  
  filtered_test <- 
    combined_dataset_plusplusplus %>%
    filter(PassengerId %in% test$PassengerId)
  
  filtered_train <- 
    combined_dataset_plusplusplus %>%
    filter(PassengerId %in% train$PassengerId)
  
  combined_workingDataset <- 
    combined_dataset_plusplusplus %>%
    group_by(Surname) %>%
    mutate(sum_Surname = n()) %>%
    mutate(frac_Surname_survived = 
             ifelse(Surname %in% filtered_test$Surname,
                    sum(Survived == 1, na.rm = TRUE)/sum(!is.na(Survived)),
                    NA)) %>%
    mutate(Surname_Train_Test =
             ifelse((Surname %in% filtered_test$Surname) &&
                      (Surname %in% filtered_train$Surname), TRUE, FALSE)) %>%
    ungroup()
  
  rm(combined_dataset, combined_dataset_plus, combined_dataset_plusplus)
  
  #bin the Age data and the Fare data
  combined_workingDataset <- 
    combined_workingDataset %>%
    mutate(Age_bin = cut(Age, c(-1, 6, 12, 21, 30, 48, 66, 100)))
  
  combined_workingDataset <- 
    combined_workingDataset %>%
    mutate(Fare_bin = cut(Fare, c(-1, 7, 8, 12, 19, 50, 78, 600)))
  
  #separate the training and the test portions
  train_extract <- 
    subset(combined_workingDataset, PassengerId %in% train$PassengerId)
  test_extract <- 
    subset(combined_workingDataset, PassengerId %in% test$PassengerId)
  
  #For the cases where the correct values for Survived may be included in the 
  #input test dataframe, restore these values to the wranged data frame.
  
  test_extract$Survived <- test_Survived
  
  return(list(train_extract, test_extract))
}

titanic_adaboost <- function(train, test, iterations, 
                             tree_depth, min_cp) {
#returns a prediction on the "test" dataset  
  num_rows <- nrow(train)
  
  #initialize the weights to be applied to each example in the training set
  train$weights <- 1/num_rows         
  
  #accuracies <- list() #??initializing the list of accuracies, measured on examples left out??
  
  models <- list() #initializing the list of rpart models
  weights <- list() #initiallizing a sum to be taken over the iterations in the for loop
  adjustment_factors <- list() #initializing a list of the factors used to adjust the example weights
  pred_list <- list()  #predictions on the test dataset from EACH tree (NOT the sum)
  for (index in 1:iterations) {
    
    #In this implementation of Adaboost, the test set is sampled with replacement, and then the 
    #weights (as prescribed in the Adaboost meta-algorithm) are used to determine the relative import
    #of the respective datapoints in an rpart implementation on the sample. I also experimented with 
    #using the weights to determine the relative probabilities for selection in the sample (I did that
    #first). However, the performance (accuracy in cross-validation) was rather disapointing, with 
    #many of the weights rather quickly converging to zero. 
    
    sample_row_num <- sample(1:num_rows, num_rows,
                        replace = TRUE) #, 
    #prob = train$weights)
    
    sample <- train[sample_row_num, ]
    
    #test_cases <- train[-sample_row_num, ]
    
    #Although frac_Surname_survived may be a useful predictor for the Titanic passengers
    #in those cases in the test set for which a value can be directly estimated (those cases 
    #in which someone with the same surname appears in the training set), the value of this 
    #variable for the examples in the training set directly reflects information about the 
    #"correct" output value for this example. For instance, for any observation in the 
    #training set, if frac_Surname_survived = 1, then we know that the value of Survived 
    #for this example is 1. This presents a circularity and overfitting problem that 
    #must be overcome if this variable is to be used. (The model may tend to be unduly focussed
    #on frac_Surname_survived.) Much of what follows is meant to address this issue. In particular, 
    #for the adaptive boosting implemented with the following code, the variable 
    #"frac_Surname_survived" will be used in only every third iteration of the boosting algorithm, 
    #beginning with the second iteration. Furthermore, at these iterations, the training set is 
    #partitioned into those elements for which the surname is also reflected in the Kaggle test set 
    #(that is, those elements for which the variable in question may be helpful in predicting 
    #the outcome in the Kaggle test dataset) and those elements for which the surname is not so 
    #reflected. The variable is thus ignored in those cases in which it is of no use. I have have
    #also, for the cases in which frac_Surname_survived will be used as a predictive variable, 
    #experiemented with the cost parameter of the rpart package.
    
    library(rpart)
    fol_1 <- formula(Survived ~ Pclass + Sex + Age_bin + Fare_bin + 
                     Title + SibSpPlusParch + Cabin_letter + 
                     Embarked + sum_Surname) 
    if (index %% 5 == 2) {
      sample_1 <- sample[ which(is.na(sample$frac_Surname_survived)), ]
      sample_2 <- sample[ which(!is.na(sample$frac_Surname_survived)), ]
      fol_2 <- formula(Survived ~ Pclass + Sex + Age_bin + Fare_bin +
                       Title + SibSpPlusParch + Cabin_letter + 
                       frac_Surname_survived + Embarked + sum_Surname)
      model_1 <- rpart(fol_1, data = sample_1, 
                       method = "class", weights = sample_1$weights,
                       control = rpart.control(maxdepth = tree_depth, #perhaps sutract 1 b/c of partitioning
                                               cp = min_cp))
      model_2 <- rpart(fol_2, data = sample_2, 
                     method = "class", weights = sample_2$weights,
                     cost = c(c_f, c_f, c_f, c_f, c_f, c_f, c_f, 1/c_f, c_f, c_f),
                     control = rpart.control(maxdepth = tree_depth-1, #sutracted 1 b/c of partitioning
                                             cp = min_cp))
      
      #Although the model has been trained on a sample (to avoid overfitting) of the training set,
      #for purposes of calcaculating the adjustments to the weights applied to the examples in 
      #the training set, we predict over the entire training set.
      train_1 <- train[ which(is.na(train$frac_Surname_survived)), ]
      train_2 <- train[ which(!is.na(train$frac_Surname_survived)), ]
      prediction_1 <- predict(model_1, newdata = train_1, type="class")
      prediction_2 <- predict(model_2, newdata = train_2, type="class")
      train_1$Prediction <- prediction_1
      train_2$Prediction <- prediction_2
      train <- rbind(train_1, train_2)
      
      #The predictions on the proper test set are calculated here:
      test_1 <- test[ which(is.na(test$frac_Surname_survived)), ]
      test_2 <- test[ which(!is.na(test$frac_Surname_survived)), ]
      pred_1 <- predict(model_1, newdata = test_1, type="class")
      pred_2 <- predict(model_2, newdata = test_2, type="class")
      test_1$Prediction <- pred_1
      test_2$Prediction <- pred_2
      test <- rbind(test_1, test_2)
      
      model <- list(model_1, model_2)   #for the purpose of examining after running
                                        #the parallel loop
      
    #the case where index is NOT divisible by 4, and so where frac_Surname_survived is not used:
    } else {
      model <- rpart(fol_1, data = sample, 
                     method = "class", weights = sample$weights,
                     control = rpart.control(maxdepth = tree_depth,
                                             cp = min_cp))
      prediction <- predict(model, newdata = train, type="class")
      
      #As already noted, we predict over the entire training set.
      train$Prediction <- prediction
      
      #The predictions on the proper test set are calculated here:
      pred <- predict(model, newdata = test, type="class")
      test$Prediction <- pred
    }
    test <- arrange(test, PassengerId)
    pred_list[[index]] <- as.numeric(test$Prediction) - 1  #stores prediction for the test set
   
    #We now handle the calculation of the adjustment factor and its application to the weights
    train <- arrange(train, PassengerId)
    #pred_list[[index]] <- as.numeric(train$Prediction) - 1 
    
    wrong_cases <- train[train$Survived != train$Prediction,]
    #sample_weights <- sample$weights
    sum_weights_misclassified <- sum(wrong_cases$weights)  #epsilon
    adjustment_factor <- 
      sqrt(sum_weights_misclassified / (1 - sum_weights_misclassified))  #beta
    correct_cases <- train[train$Survived == train$Prediction,]
    
    #apply the adjustment factors to the weights
    train <- transform(train,
                      weights = ifelse(PassengerId %in% correct_cases$PassengerId,
                                        weights * adjustment_factor, weights / adjustment_factor))
    #renormalize the weights
    train <- transform(train,
      weights = weights/sum(weights))
    
    #save the weight on the model(s) in this iteration, the model(s), and the adjustment factor
    #in a list (for both calculating the adaboost prediction and for  examing after running the
    #program
    weights[[index]] <- log((1-sum_weights_misclassified)/sum_weights_misclassified)
    models[[index]] <- model
    adjustment_factors[[index]] <- adjustment_factor
    }
  
  #Apply the weighted models to the test data, to derive our predictions
  sum_weighted_predictions <- 0 #initialize the weighted sum of the predictions 
 
  #initialize a list in which the i'th element is the the adaboost prediction for i iterations
  prediction_list <- list() 
  accuracy_list <- list()
  #confusion_matrix_list <- list()
  for (index in 1:iterations) {
    #pred <- predict(models[[index]], newdata = test, type="class")  MOVED UP 
    #weights to be added adjusted for the {0, 1} representation in the Titanic data
    pred <- pred_list[[index]] 
    sum_weighted_predictions <- 
      sum_weighted_predictions + (pred - 0.5)*weights[[index]]
    prediction_list[[index]] <- as.numeric(sum_weighted_predictions > 0)
    accuracy_list[[index]] <- mean(as.numeric(prediction_list[[index]] == test$Survived))
    #confusion_matrix_list[[index]] <- 
    #  table (pred = prediction_list[[index]], true = test$Survived)
    }
  #prediction <- as.numeric(sum_weighted_predictions > 0)
  #accuracy <- as.numeric(prediction == test$Survived)
  #mean_accuracy <- mean(accuracy)
  #confusion_matrix <- table (pred = prediction, true = test$Survived)
  
  #return a comprehensive set of information
  return(list(predictions = prediction_list, accuracies <- accuracy_list, 
              #confusion_matrices = confusion_matrix_list, 
              models = models, 
              data_frame = train, weights = weights, 
              adjustment_factors = adjustment_factors))
  }

#The beginning of the main program.
titanic_kaggle_train <- read_csv("Kaggle_train .csv")
titanic_kaggle_test <- read_csv("Kaggle_test.csv")

ada_iterations <- 60 #number of adaboost iterations
max_rpart_depth <- 2 #maximum tree depth
min_rpart_cp <- .02 #minimum improvement in "complexity" in the building of the trees
c_f <- .4 #factor for adjusting the costs associated with the variabls in the rpart models

library(doParallel)
library(foreach)

cl <- makePSOCKcluster(3)
registerDoParallel(cl)
model_info_list <- 
  foreach(index = 1:nrow(titanic_kaggle_train),  
          .packages = c("tidyverse", "stringi", 
                        "stringr", "caret")) %dopar% { 
  
  adaboost_cross_check <- titanic_kaggle_train
  
  #for leave-one-out-cross validation, each "sample" consists of one row in the data frame:
  sample_Id <- index
    
  sample_frame <- adaboost_cross_check[sample_Id, ] #the one-row "data frame"
  
  adaboost_crossTrain <- adaboost_cross_check[-sample_Id, ]
  
  titanic_kaggle_test$Survived <- NA
  
  adaboost_crossTest <- rbind(sample_frame, titanic_kaggle_test)
  
  #The data wrangling is repeated in  every iteration of the parallel loop--one repeat for each
  #element of the kaggle training set, which is "left out" once in the cross validation. As indicated
  #in the comments in the definition of the titanic_wrangle function, the value of Survived is set 
  #to NA for the left-out element, so as to avoid the pollution of the testing the data with the 
  #"correct" output. (frac_Surname_survived is calculated using the value of Survived, with the "NA"s
  #removed from the data that is averaged.)  
  wrangled <- 
    titanic_wrangle(adaboost_crossTrain, adaboost_crossTest)
  wrangled_crossTrain <- wrangled[[1]]
  wrangled_crossTest <- wrangled[[2]]
  
  #the information to be added to model_info_list, at this iteration of parallel loop:
  prediction_and_info <- 
    titanic_adaboost(wrangled_crossTrain, wrangled_crossTest, 
                     iterations = ada_iterations, 
                     tree_depth = max_rpart_depth,
                     min_cp = min_rpart_cp)
  
  }
stopCluster(cl)
rm(cl)

cross_predictions <- matrix(NA, nrow = nrow(titanic_kaggle_train),  #iitialize the prediction matrix
                            ncol = ada_iterations)
cross_accuracies <- matrix(NA, nrow = nrow(titanic_kaggle_train),  #iitialize the accuracy matrix
                            ncol = ada_iterations)
for (index in 1:nrow(titanic_kaggle_train)) {
  predictions <- model_info_list[[index]]$predictions  #"index" in this case is the passId
  for (iter in 1:ada_iterations) {
    
    #the prediction for the example at the top of the wrangled_crossTest data frame:
    cross_predictions[index, iter] <- predictions[[iter]][1]  #the prediction at the top of the d.f.
    
    cross_accuracies[index, iter] <- 
      (cross_predictions[index, iter] == titanic_kaggle_train$Survived[index]) * 1
    }
  }
averages = colMeans(cross_accuracies)

cross_confusion_list <- list()
for (index in 1:ada_iterations) {
  cross_confusion_list[[index]] <- 
    table(pred = titanic_kaggle_train$Survived, truth = cross_predictions[,index])
}

cross_confusion_list #with the list # equal to the # of iterations of the adaptive boosting

averages  #leave-one-out cross-validation averages, in the order of the # of boosting iterations
