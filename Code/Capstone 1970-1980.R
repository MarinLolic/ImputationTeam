# 2023 Capstone Project
# 1970-1980 Model


# Step 1: Setup
library(keras)
library(tensorflow)
library(dplyr)
library(fastDummies)
library(stringr)

tensorflow::set_random_seed(1)


# Step 2: Import & Clean Data
treiman <- read.csv("Treiman_File_clean.csv", header = TRUE)
treiman <- na.omit(treiman[ , 42:45])
treiman <- treiman[sample(1:nrow(treiman)), ]


# Step 3: Create One-Hot Values
X_1970_IND <- dummy_cols(treiman$ind_70_final)
X_1970_IND <- as.matrix(X_1970_IND[ , -1]) 

X_1970_OCC <- dummy_cols(treiman$occ_70_final)
X_1970_OCC <- as.matrix(X_1970_OCC[ , -1]) 

Y_1980_IND <- dummy_cols(treiman$ind_80)
Y_1980_IND <- as.matrix(Y_1980_IND[ , -1]) 

Y_1980_OCC <- dummy_cols(treiman$occ_80)
Y_1980_OCC <- as.matrix(Y_1980_OCC[ , -1]) 


# Step 4: Split Data
train_split <- 0.8
validation_split <- 0.1
test_split <- 0.1

idx1 <- round(train_split * nrow(treiman), 0)
idx2 <- round(idx1 + 1, 0)
idx3 <- round(idx2 + validation_split * nrow(treiman), 0)
idx4 <- round(idx3 + 1, 0)

X_1970_IND_Train <- X_1970_IND[1:idx1, ]
X_1970_OCC_Train <- X_1970_OCC[1:idx1, ]
Y_1980_IND_Train <- Y_1980_IND[1:idx1, ]
Y_1980_OCC_Train <- Y_1980_OCC[1:idx1, ]

X_1970_IND_Validate <- X_1970_IND[idx2:idx3, ]
X_1970_OCC_Validate <- X_1970_OCC[idx2:idx3, ]
Y_1980_IND_Validate <- Y_1980_IND[idx2:idx3, ]
Y_1980_OCC_Validate <- Y_1980_OCC[idx2:idx3, ]

X_1970_IND_Test <- X_1970_IND[idx4:nrow(X_1970_IND), ]
X_1970_OCC_Test <- X_1970_OCC[idx4:nrow(X_1970_OCC), ]
Y_1980_IND_Test <- Y_1980_IND[idx4:nrow(Y_1980_IND), ]
Y_1980_OCC_Test <- Y_1980_OCC[idx4:nrow(Y_1980_OCC), ]


# Step 5: Define Inputs
Input1970_IND <- layer_input(shape = c(ncol(X_1970_IND_Train)), name = "Input1970_IND")
Input1970_OCC <- layer_input(shape = c(ncol(X_1970_OCC_Train)), name = "Input1970_OCC")

P1 <- Input1970_IND %>% layer_dense(100, activation = "selu")
P2 <- Input1970_OCC %>% layer_dense(200, activation = "selu")


# Step 6: Define Hidden Layers
features <- layer_concatenate(list(P1, P2)) %>%
  layer_dense(300, activation = "selu") %>%
  layer_dropout(0.5)


# Step 7: Define Outputs
Output1980_IND <- features %>% layer_dense(100, activation = "selu") %>%
  layer_dense(ncol(Y_1980_IND_Train) , activation = "softmax", name = "Output1980_IND")
Output1980_OCC <- features %>% layer_dense(200, activation = "selu") %>%
  layer_dense(ncol(Y_1980_OCC_Train) , activation = "softmax", name = "Output1980_OCC")


# Step 8: Define Model
model <- keras_model(
  inputs = list(Input1970_IND, Input1970_OCC),
  outputs = list(Output1980_IND, Output1980_OCC)
)

model %>% compile(
  optimizer = "adam",
  loss = c("categorical_crossentropy", "categorical_crossentropy"),
  metrics = c("accuracy", "top_k_categorical_accuracy")
)


# Step 9: Fit Model and Evaluate
model %>% fit(
  x = list(X_1970_IND_Train, X_1970_OCC_Train),
  y = list(Y_1980_IND_Train, Y_1980_OCC_Train),
  epochs = 5
)

model %>% evaluate(
  x = list(X_1970_IND_Validate, X_1970_OCC_Validate),
  y = list(Y_1980_IND_Validate, Y_1980_OCC_Validate),
)


# Step 10: Test Set Performance
model %>% evaluate(
  x = list(X_1970_IND_Test, X_1970_OCC_Test),
  y = list(Y_1980_IND_Test, Y_1980_OCC_Test),
)


# Step 11: Impute Using 1977 Data
load(file='mar1977.Rda')
mar1977 <- mar1977[ , c("ind", "occ")]
mar1977 <- filter(mar1977, occ != 0, ind != 0)
tre2 <- treiman[ , c(1, 3)]
names(tre2) <- c("ind", "occ")
mar1977 <- rbind(mar1977, tre2)

X_1977_IND <- dummy_cols(mar1977$ind)
X_1977_IND <- as.matrix(X_1977_IND[1:74426, -1])
X_1977_IND <- X_1977_IND[ , -55]

X_1977_OCC <- dummy_cols(mar1977$occ)
X_1977_OCC <- as.matrix(X_1977_OCC[1:74426, -1])
X_1977_OCC <- X_1977_OCC[ , -c(58, 101, 303, 341, 408)]

predictions <- model %>% predict(x = list(X_1977_IND, X_1977_OCC))

Y_Impute_IND <- as.matrix(predictions[[1]])
colnames(Y_Impute_IND) <- colnames(Y_1980_IND)
Imputed_IND <- colnames(Y_Impute_IND)[max.col(Y_Impute_IND, ties.method="first")]

Y_Impute_OCC <- as.matrix(predictions[[2]])
colnames(Y_Impute_OCC) <- colnames(Y_1980_OCC)
Imputed_OCC <- colnames(Y_Impute_OCC)[max.col(Y_Impute_OCC, ties.method="first")]

Imputed_DF <- as.data.frame(cbind(word(Imputed_DF$Imputed_IND, 2, sep = "_"), 
                    word(Imputed_DF$Imputed_OCC, 2, sep = "_")))
names(Imputed_DF) <- c("imputed_IND", "imputed_OCC")

write.csv(Imputed_DF, "mar_1977_imputed.csv")
