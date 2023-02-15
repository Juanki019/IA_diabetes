################################################################################
##############               Diabetes                    #################
################################################################################


## Librerias 
library(caret) #herramientas para separacion de datos 
library(ggplot2)
library(ROCR) #curvas roc
library(grid) # for grids
library(gridExtra) # for arranging the grids

diabetes <- read.csv("diabetes.csv" , header = TRUE, sep = ",")
str(diabetes)
table(diabetes$Outcome)

#NaN
any(is.na.data.frame(diabetes))
#diabetes <- na.omit(diabetes)



################################################################################
##############               Analisis de los datos                     #################
################################################################################

#DISTRIBUCION RAZONABLE
p1 <- ggplot(diabetes, aes(x=Pregnancies)) + ggtitle("Number of times pregnant") +
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), binwidth = 1, colour="black", fill="white") + ylab("Percentage")
p2 <- ggplot(diabetes, aes(x=Glucose)) + ggtitle("Glucose") +
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), binwidth = 5, colour="black", fill="white") + ylab("Percentage")
p3 <- ggplot(diabetes, aes(x=BloodPressure)) + ggtitle("Blood Pressure") +
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), binwidth = 2, colour="black", fill="white") + ylab("Percentage")
p4 <- ggplot(diabetes, aes(x=SkinThickness)) + ggtitle("Skin Thickness") +
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), binwidth = 2, colour="black", fill="white") + ylab("Percentage")
p5 <- ggplot(diabetes, aes(x=Insulin)) + ggtitle("Insulin") +
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), binwidth = 20, colour="black", fill="white") + ylab("Percentage")
p6 <- ggplot(diabetes, aes(x=BMI)) + ggtitle("Body Mass Index") +
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), binwidth = 1, colour="black", fill="white") + ylab("Percentage")
p7 <- ggplot(diabetes, aes(x=DiabetesPedigreeFunction)) + ggtitle("Diabetes Pedigree Function") +
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), colour="black", fill="white") + ylab("Percentage")
p8 <- ggplot(diabetes, aes(x=Age)) + ggtitle("Age") +
  geom_histogram(aes(y = 100*(..count..)/sum(..count..)), binwidth=1, colour="black", fill="white") + ylab("Percentage")
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, ncol=2)


#Se puede apreciar que existe que en la matriz las variables que está coloreadas con rojo, muestran una correlación lineal fuerte, mientras que las que están con color narajo tienen una correlación intermedia a fuerte, y las que están en color ocre carne a rosado sus correlaciones lineales son débiles.
library(ggcorrplot)
matriz <- names(which(sapply(diabetes, is.numeric)))
corr <- cor(diabetes[,matriz])
ggcorrplot(corr, lab = TRUE)


#Vamos a quitar SkinThickness y BloodPressure porque segun la matriz de correlacion tienen poca relacion con tener diabetes
diabetes$BloodPressure <- NULL
diabetes$SkinThickness <- NULL

diabetes$Outcome <- as.factor(diabetes$Outcome) #para clasificar los datos
str(diabetes)

#normalizamos/limpiamos el data
preProcess_model <- preProcess(diabetes, method = 'range') #normalizamos el data  -> nos da un vector con el minimos y maximo de los data
trainData <- predict(preProcess_model, newdata = diabetes)

levels(trainData$Outcome) <- c("Class0", "Class1")  # 0 = No // 1 = Si  // Lo cambiamos el 0 a class 0...
summary(trainData)


#normalizamos/limpiamos el data
preProcess_model <- preProcess(diabetes, method = 'range') #normalizamos el data  
trainData <- predict(preProcess_model, newdata = diabetes)

#paticion random 80/20 %
trainIndex_cv <- createDataPartition(trainData$Outcome,      #output variable. createDataPartition realiza particiones proporcionales al output
                                     p = 0.8,      #training
                                     list = FALSE, #Evitar que el output sea una lista
                                     times = 1)    #una particion


levels(trainData$Outcome) <- c("Class0", "Class1")  # 0 = No // 1 = Si  // Lo cambiamos el 0 a class 0...
summary(trainData)


fitControl <- trainControl(
  method = 'cv',                     #k-fold cross-validation
  number = 8,                        #numero de divisiones
  savePredictions = 'final',         #predicciones guardadas optimas
  classProbs = T,   
  summaryFunction = twoClassSummary#resultado en summariza
)


################################################################################
##############               Logistic Regression                    #################
################################################################################

model1.fit = train(Outcome ~ ., 
                   data = trainData, 
                   method = 'glm', 
                   tuneLength = 7, 
                   trControl = fitControl)
summary(model1.fit)


################################################################################
##############               Validacion: Regresion                    #################
################################################################################


pr_Reg <- predict(model1.fit, trainData[-trainIndex_cv,])
#visualizar con la matriz de confusion el accurancy
tab <- table(trainData[-trainIndex_cv,]$Outcome, pr_Reg, dnn= c("actuales", "predicho"))
confusionMatrix(tab)
mean(pr_Reg==trainData[-trainIndex_cv,]$Outcome)


################################################################################
##############               Decission Tree                    #################
################################################################################
library(rpart)
fitControl
model2.fit <- rpart(Outcome ~ .,
                    data=trainData,
                    method="class")

summary(model2.fit)
plot(model2.fit, uniform=TRUE, 
     main="Classification Tree for Diabetes")
text(model2.fit, use.n=TRUE, all=TRUE, cex=.8)

################################################################################
##############               Validacion: Tree                    #################
################################################################################


pr <- predict(model2.fit, trainData[-trainIndex_cv,])
#visualizar con la matriz de confusion el accurancy
tab <- table(trainData[-trainIndex_cv,]$Outcome, pr, dnn= c("actuales", "predicho"))
confusionMatrix(tab)
mean(pr==trainData[-trainIndex_cv,]$Outcome)



################################################################################
##############               Classification: KNN                    #################
################################################################################


model3.fit = train(Outcome ~ .,
                   data = trainData[trainIndex_cv,], 
                   method = 'knn', 
                   tuneLength = 5, 
                   trControl = fitControl)#KNN

model3.fit
ggplot(model3.fit) 

################################################################################
##############               Validacion: KNN                    #################
################################################################################


pr_KNN <- predict(model3.fit, trainData[-trainIndex_cv,])
tab <- table(trainData[-trainIndex_cv,]$Outcome, pr_KNN, dnn= c("actuales", "predicho"))
confusionMatrix(tab)
mean(pr_KNN==trainData[-trainIndex_cv,]$Outcome)


################################################################################
##############               Classification: SVM                    #################
################################################################################


model5.fit = train(Outcome ~ .,
                   data = trainData[trainIndex_cv,], 
                   method = 'svmRadial', 
                   tuneLength = 5, 
                   trControl = fitControl)#KNN

model5.fit
ggplot(model5.fit) 

################################################################################
##############               Validacion: SVM                    #################
################################################################################


pr_SVM <- predict(model5.fit, trainData[-trainIndex_cv,])
#visualizar con la matriz de confusion el accurancy
tab <- table(trainData[-trainIndex_cv,]$Outcome, pr_SVM, dnn= c("actuales", "predicho"))
confusionMatrix(tab)
mean(pr_SVM==trainData[-trainIndex_cv,]$Outcome)



################################################################################
##############               Classification: Bayes                    #################
################################################################################
library(e1071) # for naive bayes
model4.fit <- naiveBayes(Outcome ~ ., 
                         data = trainData[trainIndex_cv,], 
                         tuneLength = 7, 
                         trControl = fitControl)  #toma los indices del train
model4.fit


################################################################################
##############               Validacion: Bayes                    #################
################################################################################

pr_Bayes <- predict(model4.fit, trainData[-trainIndex_cv,])
#visualizar con la matriz de confusion el accurancy
tab <- table(trainData[-trainIndex_cv,]$Outcome, pr_Bayes, dnn= c("actuales", "predicho"))
confusionMatrix(tab)
mean(pr_Bayes==trainData[-trainIndex_cv,]$Outcome)


models_compare <- resamples(list(GLM=model1.fit, fitKNN=model3.fit, SVM=model5.fit))

summary(models_compare)

dibujo <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales = dibujo)

