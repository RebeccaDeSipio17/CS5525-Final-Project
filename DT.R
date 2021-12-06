# Using Decision Tree
# Using the heart.csv data set
# Here, the target column indicates whether a person has heart disease or not

# setwd("C:/Users/Francis/OneDrive/PhD/Data Analytics I/Final Project")

setwd("/Users/ajinkyafotedar/CS5525/Project/CS5525-Final-Project")
set.seed(2441139)
library(tree)

# Read data
heart <- read.csv("heart.csv")
Target <- as.factor(heart$target)

# Split into training and test sets
train <- sample(1:nrow(heart), 0.75*nrow(heart))
heart.test <- heart[-train, ]
Target.test <- Target[-train]

# Get tree
tree.heart <- tree(Target~. -target, heart, subset=train)
summary(tree.heart)

# Plot tree
plot(tree.heart)
text(tree.heart, pretty=1, cex=0.7)

# Predict using test set
tree.pred <- predict(tree.heart, heart.test, type='class')
table(tree.pred, Target.test)

# Now lets prune
set.seed(2441139)
cv.heart <- cv.tree(tree.heart, FUN=prune.misclass)
cv.heart
plot(cv.heart$size, cv.heart$dev, type='b')
prune.heart <- prune.misclass(tree.heart, best=10)
plot(prune.heart)
text(prune.heart, pretty=1, cex=0.65)

# Predict using pruned tree
prune.pred <- predict(prune.heart, heart.test, type='class')
table(prune.pred, Target.test)


# Bagging
library(randomForest)
set.seed(2441139)
bag.heart <- randomForest(as.factor(as.character(heart$target))~., data=heart, 
                          subset=train, mtry=ncol(heart)-1, 
                          importance=TRUE)
bag.heart

# Predict on bagged tree
bag.pred <- predict(bag.heart, heart.test, type='class')
table(bag.pred, Target.test)
varImpPlot(bag.heart)


# Random Forest
# Here, we could investigate how mtry and ntree affect the accuracy
set.seed(2441139)
rf.heart <- randomForest(as.factor(as.character(heart$target))~., data=heart, 
                          subset=train, mtry=sqrt(ncol(heart)-1), 
                          ntree=25, importance=TRUE)
rf.heart

# Predict on the forest
rf.pred <- predict(rf.heart, heart.test, type='class')
table(rf.pred, Target.test)
varImpPlot(rf.heart)

## Determine best model 
# Here, we could investigate how mtry affect the accuracy
Acc <- rep(0,ncol(heart)-2)
for (m in 1:(ncol(heart)-2)){
  set.seed(2441139)
  rf.heart <- randomForest(as.factor(as.character(heart$target))~., data=heart, 
                           subset=train, mtry=m, 
                           ntree=25)
  rf.pred <- predict(rf.heart, heart.test, type='class')
  t <- table(rf.pred, Target.test)
  acc <- sum(diag(t))/sum(t)
  Acc[m] <- acc
}
mbest <- which(Acc==max(Acc))
plot(1:(ncol(heart)-2), Acc, xlab='mtry', ylab='Accuracy of random forest')


# Now use the best value of m for the random forest
set.seed(2441139)
rf.heart <- randomForest(as.factor(as.character(heart$target))~., data=heart, 
                         subset=train, mtry=mbest, 
                         ntree=25, importance=TRUE)
rf.heart

# Predict on the forest
rf.pred <- predict(rf.heart, heart.test, type='class')
table(rf.pred, Target.test)
varImpPlot(rf.heart)



##########################################################
# Logistic Regression
log.heart <- glm(target~., data=heart, family=binomial, subset=train)
summary(log.heart)
exp(coef(log.heart))

# Here, we can plot the log regression with each significant variable
plot(heart$target, heart$age)
plot(heart$target, heart$oldpeak)
plot(heart$target, heart$fbs)

# Predictions
log.probs <- predict(log.heart, heart.test, type='response')
log.pred <- rep("No", nrow(heart.test))
log.pred[log.probs > 0.5] <- "Yes"
table(log.pred, Target.test)



##########################################################
# K-Nearest Neighbor
library(class)
cl <- as.factor(heart$target[train])
knn.heart <- knn(heart[train,], heart.test, cl, k = 5, prob=TRUE)
table(knn.heart, Target.test)
