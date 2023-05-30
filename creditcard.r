
## Loading all the necesssary libraries
library(tidyverse)
install.packages("gridExtra")
library(caret)
library(e1071)
library(MASS)
library(ggplot2)
library(randomForest)
library(glmnet)
library(gbm)
library(smotefamily)
library(performanceEstimation)
library(class)
library(gridExtra)
library(tree)
library(randomForest)
library(reshape2)

#  Reading the datasets available
bank_additional <- read.table("Data/bank-additional.csv",header=TRUE, sep=";")
view(bank_additional)
bank_additional_full<- read.table("Data/bank-additional-full.csv",header=TRUE, sep=";")

#Looking at the datasets 
view(bank_additional_full)
summary(bank_additional_full)



set.seed(5)


#Tidying the data

#Removing Unknown values  
bank_additional_full<- bank_additional_full %>%
  filter(job !="unknown" & marital != "unknown" & education !="unknown"& default !="unknown"& loan !="unknown"& housing !="unknown") 





#Creating a new dataset to find the corelation matrix
bank_additional_full_numeric<-bank_additional_full[c( 1,11,12,13,14,16,17,18,19,20,21)]
bank_additional_full_numeric$y<- as.numeric(bank_additional_full_numeric$y=="yes")
unique(bank_additional_full_numeric$y)
view(bank_additional_full_numeric)
bank_additional_full_numeric
cor_mat<-round(cor(bank_additional_full_numeric),2)
melted_corr_mat <- melt(cor_mat)

#Plotting the correlation Matrix
ggplot(data = melted_corr_mat, aes(x=Var1, y=Var2,fill=value)) +
  geom_tile() +
  geom_text(aes(Var2, Var1, label = value),color = "black", size = 4)




##We can see that pdays and previous are corelated with each other and we need to remove one of them.
## We will be removing previous as pdays has stronger corealtion with y.

bank_additional_full<- subset(bank_additional_full, select = -c(previous))






## PLotting bargraph of the data to have a goood view of the data./

## Plotting for job
plot_job<- ggplot(data = bank_additional_full, mapping = aes(job)) +
  geom_bar(aes(fill=y), position = position_dodge())

plot_job

## Plotting for education
plot_education<- ggplot(data = bank_additional_full, mapping = aes(education)) +
  geom_bar(aes(fill=y), position = position_dodge())

plot_education

##Plotting for marital status
plot_marital<- ggplot(data = bank_additional_full, mapping = aes(marital)) +
  geom_bar(aes(fill=y), position = position_dodge())

plot_marital

##Plotting for default status
plot_default<- ggplot(data = bank_additional_full, mapping = aes(default)) +
  geom_bar(aes(fill=y), position = position_dodge())

plot_default

##Plotting for loan status
plot_loan<- ggplot(data = bank_additional_full, mapping = aes(loan)) +
  geom_bar(aes(fill=y), position = position_dodge())

plot_loan

##Plotting for contact status
plot_contact<- ggplot(data = bank_additional_full, mapping = aes(contact)) +
  geom_bar(aes(fill=y), position = position_dodge())

plot_contact

##Plotting for day_of_week status
plot_day_of_week<- ggplot(data = bank_additional_full, mapping = aes(day_of_week)) +
  geom_bar(aes(fill=y), position = position_dodge())

plot_day_of_week

##Plotting for month status
plot_month<- ggplot(data = bank_additional_full, mapping = aes(month)) +
  geom_bar(aes(fill=y), position = position_dodge())

plot_month


##Plotting for poutcome status
plot_poutcome<- ggplot(data = bank_additional_full, mapping = aes(poutcome)) +
  geom_bar(aes(fill=y), position = position_dodge())





##From the bar plots, we are only selecting poutcome, loan, default and housing.
bank_additional_full<- bank_additional_full<- subset(bank_additional_full, select = -c(day_of_week, month,education, marital,job,contact))
bank_additional_full$y<- as.numeric(bank_additional_full$y=="yes")
bank_additional_full$housing<- as.numeric(bank_additional_full$housing=="yes")
bank_additional_full$loan<- as.numeric(bank_additional_full$loan=="yes")
bank_additional_full$default<- as.numeric(bank_additional_full$default=="yes")
bank_additional_full$poutcome <- as.numeric(factor(bank_additional_full$poutcome ))
view(bank_additional_full)



#plotting classes of response variable in pie chart
n<- table(bank_additional_full$y)
count<- data.frame(y= c(0,1), n=c(26629, 3859))
count
plot_y<-ggplot(count,aes(  x="",y=n, fill=y)) +
  geom_col( width = 1)+
  coord_polar(theta ="y")+
  theme_classic()

plot_y






set.seed(10)

index<- sample(1:nrow(bank_additional_full), nrow(bank_additional_full)/2)
train_bank_additional_full<-bank_additional_full[index, ]
test_bank_additional_full<-bank_additional_full[-index, ]



## logisitic Regression
y_logisitic <- glm(y~.,family = "binomial", data=train_bank_additional_full)
summary(y_logisitic)


names(y_logisitic)
y_predictions_logisitic = predict(y_logisitic, test_bank_additional_full, type="response")

y_predictions_logisitic = data.frame(y=y_predictions_logisitic)

y_predictions_logisitic = mutate(y_predictions_logisitic, y = ifelse(y >= 0.50, 1, 0))
head(y_predictions_logisitic)
confusionMatrix(as.factor(y_predictions_logisitic$y) , as.factor(test_bank_additional_full$y))
view(test_bank_additional_full$y)







## Since the Pr(>|z) value of  default, housing, loan,euribor3m is more than 0.1 we will try to fit the data by removing them.
train_bank_additional_full_2<- subset(train_bank_additional_full, select=-c( default,housing, euribor3m))
test_bank_additional_full_2<- subset(test_bank_additional_full, select=-c(default,housing, euribor3m))

y_logisitic_2 <- glm(y~.,family = "binomial", data=train_bank_additional_full_2)
summary(y_logisitic_2)
y_predictions_logisitic_2 = predict(y_logisitic_2, test_bank_additional_full_2, type="response")

y_predictions_logisitic_2 = data.frame(y=y_predictions_logisitic_2)

y_predictions_logisitic_2 = mutate(y_predictions_logisitic_2, y = ifelse(y >= 0.50, 1, 0))
head(y_predictions_logisitic_2)
confusionMatrix(as.factor(y_predictions_logisitic_2$y) , as.factor(test_bank_additional_full_2$y))

##It improves the accuracy by a little bit.




##lda

lda_Y <- lda(y~., data=train_bank_additional_full)
lda_Y
plot(lda_Y)
lda_predictions<- predict(lda_Y,test_bank_additional_full)
confusionMatrix(lda_predictions$class,as.factor(test_bank_additional_full$y))


lda_Y_2 <- lda(y~., data=train_bank_additional_full_2)
lda_Y_2
plot(lda_Y_2)
lda_predictions_2<- predict(lda_Y_2,test_bank_additional_full_2)
confusionMatrix(lda_predictions_2$class,as.factor(test_bank_additional_full_2$y))
## Improves the accuracy slightly

###bank_additional_full - QDA
qda_Y_2 = qda(y~., data=train_bank_additional_full_2)
qda_Y_2
qda_predictions_2<- predict(qda_Y_2,test_bank_additional_full_2)
confusionMatrix(as.factor(qda_predictions_2$class) , as.factor(test_bank_additional_full_2$y))









###KNN (K NEARESt NEighbours)

#Data normalistion to avoid biasness as the value sclae of 'nr.employed' , 'duration' is in thousand whereas other attribute's value are in 2 digits or 1 digit.
normalize <- function(x) {return ((x - min(x)) / (max(x) - min(x)))}


#Data normalistion to avoid biasness as the value sclae of 'Credit.Amount'is in thousand whereas other attribute's value are in 2 digits or 1 digit.
bank_knn_data<- as.data.frame(lapply(train_bank_additional_full[,1:13], normalize))


view(bank_knn_data)

#Creating Training and Test data set. Training data will be used to build model whereas test data will be used for validation and optimisation of model by tuning k value.
train_bank_knn_data<- bank_additional_full[index, ]
test_bank_knn_data<- bank_additional_full[-index,]
train_bank_knn_labels <- bank_additional_full[index, 14]
test_bank_knn_labels  <- bank_additional_full[-index,14]
view(test_bank_knn_labels)

optimum_K<- sqrt(nrow(train_bank_knn_data))
optimum_K ## Since the square root of the number of rows is 123.46, the optimum k will be within that range)


i=1                         # declaration to initiate for loop
k.optm=1                    # declaration to initiate for loop
for (i in seq(1, 150, by=5)){ 
  knn.mod <-  knn(train=train_bank_additional_full, test=test_bank_additional_full, cl=train_bank_knn_labels, k=i)
  k.optm[i] <- 100 * sum(test_bank_knn_labels == knn.mod)/NROW(test_bank_knn_labels)
  k=i  
  cat(k,'=',k.optm[i],'\n')       # to print % accuracy 
}

plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level")

which.max(k.optm[])
#### At k=81, maximum accuracy achieved which is 90.2191%, after that, it seems increasing K increases the classification but reduces success rate. 


## Applying knn for k=81
knn.81 <-  knn(train=train_bank_knn_data, test=test_bank_knn_data, cl=train_bank_knn_labels, k=81)
knn.81
test_bank_knn_labels
confusionMatrix(as.factor(knn.81) ,as.factor(test_bank_knn_labels))









##### Decision Trees







## Fititng a decision tree on training data.
Y_Decision_tree <- tree(y ~ ., data=bank_additional_full)
plot(Y_Decision_tree)
text(Y_Decision_tree,pretty = 0)
Y_tree_predictions = predict(Y_Decision_tree, test_bank_additional_full)
Y_tree_predictions = data.frame(y=Y_tree_predictions)
Y_tree_predictions = mutate(Y_tree_predictions, y = ifelse(y >= 0.50, 1, 0))
confusionMatrix(as.factor(Y_tree_predictions$y), as.factor(test_bank_additional_full$y))

# The cv.tree function performs cross-validation to determine the optimal tree complexity
Y_Decision_tree_cv = cv.tree(Y_Decision_tree)
Y_Decision_tree
# The object contains different terminal node values, their error rate, and cost complexity parameter
Y_Decision_tree_cv_df = data.frame(Nodes=Y_Decision_tree_cv$size,
                                 Error=Y_Decision_tree_cv$dev,
                                 Alpha=Y_Decision_tree_cv$k)

# Plot the number of terminal nodes, and their corresponding errors and alpha parameters
Y_Decision_tree_cv_error = ggplot(Y_Decision_tree_cv_df, aes(x=Nodes, y=Error)) + geom_line() + geom_point()
Y_Decision_tree_cv_alpha = ggplot(Y_Decision_tree_cv_df, aes(x=Nodes, y=Alpha)) + geom_line() + geom_point()


# Show the plots side-by-side with the grid.arrange function from gridExtra package
grid.arrange(Y_Decision_tree_cv_error,Y_Decision_tree_cv_alpha, ncol=2)

# A tree with 7 terminal nodes results in the lowest error
# This also corresponds to alpha value of 0

# Finally, prune the tree with prune.misclass function and specify 7 terminal nodes
Y_Decision_tree_pruned = prune.tree(Y_Decision_tree, best=7)

# Plot the pruned tree
plot(Y_Decision_tree_pruned)
text(Y_Decision_tree_pruned, pretty=0)

# Use the pruned tree to make predictions, and compare the accuracy to the non-pruned tree
Y_tree_pruned_predictions = predict(Y_Decision_tree_pruned, test_bank_additional_full)
Y_tree_pruned_predictions = data.frame(y=Y_tree_pruned_predictions)
Y_tree_pruned_predictions = mutate(Y_tree_pruned_predictions, y = ifelse(y >= 0.50, 1, 0))
confusionMatrix(as.factor(Y_tree_pruned_predictions$y), as.factor(test_bank_additional_full$y))


# Pruning results in the same accuracy for the  model.

### BAGGUING

train_bank_additional_full$y <- as.factor(train_bank_additional_full$y)
view(train_bank_additional_full)
Y_bag = randomForest(y ~ ., data=train_bank_additional_full, mtry=13, importance=TRUE)
Y_bag
view(Y_bag)

Y_bag_predictions<- predict(Y_bag, test_bank_additional_full)
view(Y_bag_predictions)
Y_bag_predictions = data.frame(y=Y_bag_predictions)
confusionMatrix(as.factor(Y_bag_predictions$y), as.factor(test_bank_additional_full$y))

##RandomTree

Y_rf<- randomForest(y~., data = train_bank_additional_full, mtry=sqrt(13), importance=TRUE)
Y_rf_predictions<- predict(Y_rf , test_bank_additional_full)
Y_rf_predictions = data.frame(y=Y_rf_predictions)
confusionMatrix(as.factor(Y_rf_predictions$y), as.factor(test_bank_additional_full$y))
plot(Y_rf)
importance(Y_rf)
varImpPlot(Y_rf)









####SVM
#Using a smaller dataset which was sampled from the previous one to fit the SVM
#ALSO repeating the steps for data analysis and tidying
bank_additional <- read.table("Data/bank-additional.csv",header=TRUE, sep=";")
bank_additional<- bank_additional %>%
  filter(job !="unknown" & marital != "unknown" & education !="unknown"& default !="unknown"& loan !="unknown"& housing !="unknown") 
bank_additional<- subset(bank_additional_full, select = -c(previous))

bank_additional <- subset(bank_additional, select = -c(day_of_week, month,education, marital,job,contact,default))
bank_additional$housing<- as.numeric(bank_additional$housing=="yes")
bank_additional$loan<- as.numeric(bank_additional$loan=="yes")
bank_additional$default<- as.numeric(bank_additional$default=="yes")
bank_additional$poutcome <- as.numeric(factor(bank_additional$poutcome ))
view(bank_additional)

set.seed(10)

index<- sample(1:nrow(bank_additional), nrow(bank_additional)/2)
train_bank_additional<-bank_additional[index, ]
test_bank_additional<-bank_additional[-index, ]



train_bank_additional$y <- as.factor(train_bank_additional$y)

#Find the best linear kernel
tune_lin <- tune(svm, y~ ., data = train_bank_additional, kernel = "linear",ranges = list(cost = c(1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2)))
best_lin <- tune_lin$best.model
best_lin
svm.bestFit_linear <- svm(y ~., data = train_bank_additional, kernel = "linear", cost = 10)

svm.test.pred <- predict(svm.bestFit_linear, test_bank_additional)
svm.test.pred = data.frame(y=svm.test.pred)
confusionMatrix(as.factor(svm.test.pred$y), as.factor(test_bank_additional$y))
# Find best polynomial kernel.
tune_pol <- tune(svm, y ~ ., data = train_bank_additional, kernel = "polynomial",ranges = list(cost = c(1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2),
                               degree = c(2, 3, 4, 5)))
best_pol <- tune_pol$best.model
best_pol

svm.bestFit_poly <- svm(y ~., data = train_bank_additional, kernel = "polynomial", cost = 10, degree=3)


svm.test.pred <- predict(svm.bestFit_poly, test_bank_additional)
svm.test.pred <- data.frame(y=svm.test.pred)
confusionMatrix(as.factor(svm.test.pred$y), as.factor(test_bank_additional$y))

# Find best radial kernel.
tune_rad <- tune(svm, y ~ ., data = train_bank_additional, kernel = "radial",ranges = list(cost = c(1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2),gamma = c(0.5, 1, 2, 3, 4)))
best_rad <- tune_rad$best.model
best_rad
svm.test.pred <- predict(best_rad, test_bank_additional)
svm.test.pred = data.frame(y=svm.test.pred)
confusionMatrix(as.factor(svm.test.pred$y), as.factor(test_bank_additional$y))


## Among the SVMs, the linear kernel has the best accuracy.



## Plotting the values from the models in a bar graph.

all_results<- data.frame(models=c("LR","LDA","QDA","kNN","Bagging","RF","DecisionTree","SVM"),Accuracy=c(0.8967,0.8939, 0.8708, 0.9023,0.8998,0.8998,  0.8941,0.8971), Sensitivity=c(0.9758,0.9667,0.9112,0.9651,0.9529,0.9616,0.9640,0.9784), Specificity=c(0.3664,0.4073,0.5938,0.4824,0.5449,0.5171,0.4269,0.2680),PPV=c(0.9115,0.9160,0.9375,0.9257,0.9333,0.9301,0.9183,0.9038),NPV=c(0.6964,0.6464,0.5030,0.6739,0.6340,0.6684,0.6392,0.7536))


ggplot(aes(x = reorder(models, Accuracy),y = Accuracy),data = all_results) +
  geom_bar(stat = 'identity', aes(fill =models) )+
  geom_text(aes(label = round(Accuracy,4))) + theme_bw() + 
  theme(axis.text.x = element_text(angle = 40))


ggplot(aes(x = reorder(models, Sensitivity),y = Sensitivity),data = all_results) +
  geom_bar(stat = 'identity', aes(fill =models) )+
  geom_text(aes(label = round(Sensitivity,4))) + theme_bw() + 
  theme(axis.text.x = element_text(angle = 40))

ggplot(aes(x = reorder(models, Specificity),y = Specificity),data = all_results) +
  geom_bar(stat = 'identity', aes(fill =models) )+
  geom_text(aes(label = round(Specificity,4))) + theme_bw() + 
  theme(axis.text.x = element_text(angle = 40))

ggplot(aes(x = reorder(models, PPV),y = PPV),data = all_results) +
  geom_bar(stat = 'identity', aes(fill =models) )+
  geom_text(aes(label = round(PPV,4))) + theme_bw() + 
  theme(axis.text.x = element_text(angle = 40))

ggplot(aes(x = reorder(models, NPV),y = NPV),data = all_results) +
  geom_bar(stat = 'identity', aes(fill =models) )+
  geom_text(aes(label = round(NPV,4))) + theme_bw() + 
  theme(axis.text.x = element_text(angle = 40))
