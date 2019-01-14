# This is a rigorous analysis on the famous "Red-Wine" dataset, where the objective is to determine the quality level of wine
# based on the factors given. One of the biggest advantage of this datset is that it does not have any mising values, which makes
# the analysis a bit easier.
#
# Unlike other approaches, where classification makes it easier to formulate a model with great results, I tried to develop models
# that can predict the quality level directly, in stead of classifying the wines as good or bad or othe classifications. I developed 
# two models: 1. Proportional Odds model, 2. multinomial logistics regression model; both of the models can predict categorical
# outcomes with more than two possible results. There are six possible outcomes in this datset: 3,4,5,6,7,8, whch depict different 
# qulaity levels of red wine; and my models can predict with an accuracy of ~64%.

library(zoo)
library(lmtest)
library(dplyr)
library(MASS)
library(ggplot2)
library(foreign)
library(Hmisc)
library(reshape2)
library(lazyeval)
library(corrgram)
library(nnet)

# Extracting the datset
wine_data <- read.csv("winequality-red.csv", header = TRUE)
attach(wine_data)


# Checking if the datset have missing values
sapply(wine_data, function(x)all(any(is.na(x))))
lapply(wine_data, class)

# Creating the training and validation dataset
train_wine_data <- wine_data[1:1300, ]
valid_wine_data <- wine_data[1301:dim(wine_data)[1], ]
train_wine_data$quality <- as.factor(train_wine_data$quality)

# COrrelation calculation for later use
rcorr(as.matrix(train_wine_data),type = ("spearman"))
corrgram(wine_data, lower.panel=panel.shade, upper.panel=panel.ellipse)

#--------------Proportional Odss model generation---------------------------------------------

m.t3 <- polr(quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide 
            + total.sulfur.dioxide + density + pH + sulphates + alcohol, data = train_wine_data, Hess = TRUE)
summary(m.t3)

(ctable <- coef(summary(m.t3)))
p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
(ctable <- cbind(ctable, "p value" = p))
(ci <- confint.lm(m.t3))

valid_wine_data <- cbind(valid_wine_data, predict(m.t3, valid_wine_data, type = "probs"))
head(valid_wine_data)

x <- 12 + apply(valid_wine_data[,(ncol(valid_wine_data)-5):ncol(valid_wine_data)],1,which.max)
pred <- names(valid_wine_data)[x]
valid_wine_data <- cbind(valid_wine_data, pred)
Accuracy <- sum(ifelse(pred == valid_wine_data$quality,1,0))/nrow(valid_wine_data) 
Accuracy

#---------------Proportional Odds Model with correlation analysis---------------------------------------------

m.t4 <- polr(quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide 
             + total.sulfur.dioxide + density + pH + sulphates + alcohol + fixed.acidity*citric.acid + fixed.acidity*density +
             fixed.acidity*pH + citric.acid*chlorides + citric.acid*density + citric.acid*pH + density*pH + density*alcohol +
             pH*sulphates, data = train_wine_data, Hess = TRUE)
summary(m.t4)

(ctable <- coef(summary(m.t4)))
p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
(ctable <- cbind(ctable, "p value" = p))
(ci <- confint.lm(m.t4))

valid_wine_data <- wine_data[1301:dim(wine_data)[1], ]
valid_wine_data <- cbind(valid_wine_data, predict(m.t4, valid_wine_data, type = "probs"))
head(valid_wine_data)

x <- 12 + apply(valid_wine_data[,(ncol(valid_wine_data)-5):ncol(valid_wine_data)],1,which.max)
pred <- names(valid_wine_data)[x]
valid_wine_data <- cbind(valid_wine_data, pred)
Accuracy <- sum(ifelse(pred == valid_wine_data$quality,1,0))/nrow(valid_wine_data) 
Accuracy

#-------------Proportional Odds Model with statistically significant factors------------------------------------

m.t5 <- polr(quality ~ fixed.acidity + volatile.acidity + citric.acid + density + pH + sulphates + alcohol + fixed.acidity*density +
               fixed.acidity*pH + citric.acid*density + citric.acid*pH + density*pH + density*alcohol +
               pH*sulphates, data = train_wine_data, Hess = TRUE)
summary(m.t5)

(ctable <- coef(summary(m.t5)))
p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
(ctable <- cbind(ctable, "p value" = p))
(ci <- confint.lm(m.t4))

valid_wine_data <- wine_data[1301:dim(wine_data)[1], ]
valid_wine_data <- cbind(valid_wine_data, predict(m.t5, valid_wine_data, type = "probs"))
head(valid_wine_data)

x <- 12 + apply(valid_wine_data[,(ncol(valid_wine_data)-5):ncol(valid_wine_data)],1,which.max)
pred <- names(valid_wine_data)[x]
valid_wine_data <- cbind(valid_wine_data, pred)
Accuracy <- sum(ifelse(pred == valid_wine_data$quality,1,0))/nrow(valid_wine_data) 
Accuracy

#-----------------Multinomial Logistics Regression Model-----------------------------------------------

m.t6 <- multinom(quality ~ fixed.acidity + volatile.acidity + citric.acid + density + pH + sulphates + alcohol + fixed.acidity*density +
               fixed.acidity*pH + citric.acid*density + citric.acid*pH + density*pH + density*alcohol +
               pH*sulphates, data = train_wine_data)
summary(m.t6)

valid_wine_data <- wine_data[1301:dim(wine_data)[1], ]
valid_wine_data <- cbind(valid_wine_data, predict(m.t6, valid_wine_data, type = "probs"))
head(valid_wine_data)

x <- 12 + apply(valid_wine_data[,(ncol(valid_wine_data)-5):ncol(valid_wine_data)],1,which.max)
pred <- names(valid_wine_data)[x]
valid_wine_data <- cbind(valid_wine_data, pred)
Accuracy <- sum(ifelse(pred == valid_wine_data$quality,1,0))/nrow(valid_wine_data) 
Accuracy
