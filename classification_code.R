# SVM Classification Method

set.seed(2441139)
library(e1071)

# read data
heart <- read.csv("heart.csv")
Target <- as.factor(heart$target)

# split into training and test data
train <- sample(1:nrow(heart), 0.75*nrow(heart))
heart.train <- heart[train, ]
heart.test <- heart[-train, ]
Target.train <- Target[train]
Target.test <- Target[-train]

dat <- data.frame(
  x=heart.train,
  y=as.factor(Target.train)
)
# SVM
out <- svm(y ~., data=dat, kernal="radial", cost=10)
summary(out)

table(out$fitted, dat$y)

# performance on test observations
dat.test <- data.frame(
  x = heart.test,
  y = as.factor(Target.test)
)

pred.test <- predict(out, newdata = dat.test)
table(pred.test, dat.test$y)
