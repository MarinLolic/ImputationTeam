# 2023 Capstone Project
# Occupation Model


# Step 1: Setup
library(keras)
library(tensorflow)
library(dplyr)
library(fastDummies)

tensorflow::set_random_seed(1)


# Step 2: Import & Clean Data
occupation <- read.csv("OCC_crosswalk_FULL.csv", header = TRUE)
occupation <- na.omit(occupation[1:nrow(occupation), 2:8])
occupation <- occupation[sample(1:nrow(occupation)), ]


# Step 3: Create X and Y Values
X1 <- X2 <- X3 <- X4 <- X5 <- X6 <- X7 <- X8 <- X9 <- X10 <- X11 <- X12 <- occupation
for (row in 1:nrow(X2)) {X2[row, sample(seq(7), sample(seq(6), 1))] <- 0}
for (row in 1:nrow(X3)) {X3[row, sample(seq(7), sample(seq(6), 1))] <- 0}
for (row in 1:nrow(X4)) {X4[row, sample(seq(7), sample(seq(6), 1))] <- 0}
for (row in 1:nrow(X5)) {X5[row, sample(seq(7), sample(seq(6), 1))] <- 0}
for (row in 1:nrow(X6)) {X6[row, sample(seq(7), sample(seq(6), 1))] <- 0}
for (row in 1:nrow(X8)) {X8[row, sample(seq(7), sample(seq(6), 1))] <- 0}
for (row in 1:nrow(X9)) {X9[row, sample(seq(7), sample(seq(6), 1))] <- 0}
for (row in 1:nrow(X10)) {X10[row, sample(seq(7), sample(seq(6), 1))] <- 0}
for (row in 1:nrow(X11)) {X11[row, sample(seq(7), sample(seq(6), 1))] <- 0}
for (row in 1:nrow(X12)) {X12[row, sample(seq(7), sample(seq(6), 1))] <- 0}
X_full <- rbind(X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12)

Y1 <- Y2 <- Y3 <- Y4 <- Y5 <- Y6 <- Y7 <- Y8 <- Y9 <- Y10 <- Y11 <- Y12 <- occupation
Y_full <- rbind(Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Y12)


# Step 4: Create One-Hot Values
X_1972 <- dummy_cols(X_full$OCC.1972.1982)
X_1972 <- as.matrix(X_1972[ , -1]) 
X_1983 <- dummy_cols(X_full$OCC.1983.1991)
X_1983 <- as.matrix(X_1983[ , -1]) 
X_1992 <- dummy_cols(X_full$OCC.1992.Aug1995)
X_1992 <- as.matrix(X_1992[ , -1]) 
X_1995 <- dummy_cols(X_full$OCC.Sept1995.2002)
X_1995 <- as.matrix(X_1995[ , -1]) 
X_2003 <- dummy_cols(X_full$OCC.2003.2010)
X_2003 <- as.matrix(X_2003[ , -1]) 
X_2011 <- dummy_cols(X_full$OCC.2011.2019)
X_2011 <- as.matrix(X_2011[ , -1]) 
X_2020 <- dummy_cols(X_full$OCC.2020.)
X_2020 <- as.matrix(X_2020[ , -1]) 

zvec <- rep(0, nrow(Y_full))

Y_1972 <- dummy_cols(Y_full$OCC.1972.1982)
Y_1972 <- as.matrix(Y_1972[ , -1]) 
Y_1972 <- cbind(zvec, Y_1972)
Y_1983 <- dummy_cols(Y_full$OCC.1983.1991)
Y_1983 <- as.matrix(Y_1983[ , -1]) 
Y_1983 <- cbind(zvec, Y_1983)
Y_1992 <- dummy_cols(Y_full$OCC.1992.Aug1995)
Y_1992 <- as.matrix(Y_1992[ , -1]) 
Y_1992 <- cbind(zvec, Y_1992)
Y_1995 <- dummy_cols(Y_full$OCC.Sept1995.2002)
Y_1995 <- as.matrix(Y_1995[ , -1]) 
Y_1995 <- cbind(zvec, Y_1995)
Y_2003 <- dummy_cols(Y_full$OCC.2003.2010)
Y_2003 <- as.matrix(Y_2003[ , -1]) 
Y_2003 <- cbind(zvec, Y_2003)
Y_2011 <- dummy_cols(Y_full$OCC.2011.2019)
Y_2011 <- as.matrix(Y_2011[ , -1]) 
Y_2011 <- cbind(zvec, Y_2011)
Y_2020 <- dummy_cols(Y_full$OCC.2020.)
Y_2020 <- as.matrix(Y_2020[ , -1]) 
Y_2020 <- cbind(zvec, Y_2020)


# Step 5: Split Data
train_split <- 0.8
validation_split <- 0.1
test_split <- 0.1

idx1 <- round(train_split * nrow(X_full), 0)
idx2 <- round(idx1 + 1, 0)
idx3 <- round(idx2 + validation_split * nrow(X_full), 0)
idx4 <- round(idx3 + 1, 0)

X_1972_Train <- X_1972[1:idx1, ]
X_1983_Train <- X_1983[1:idx1, ]
X_1992_Train <- X_1992[1:idx1, ]
X_1995_Train <- X_1995[1:idx1, ]
X_2003_Train <- X_2003[1:idx1, ]
X_2011_Train <- X_2011[1:idx1, ]
X_2020_Train <- X_2020[1:idx1, ]

Y_1972_Train <- Y_1972[1:idx1, ]
Y_1983_Train <- Y_1983[1:idx1, ]
Y_1992_Train <- Y_1992[1:idx1, ]
Y_1995_Train <- Y_1995[1:idx1, ]
Y_2003_Train <- Y_2003[1:idx1, ]
Y_2011_Train <- Y_2011[1:idx1, ]
Y_2020_Train <- Y_2020[1:idx1, ]

X_1972_Validate <- X_1972[idx2:idx3, ]
X_1983_Validate <- X_1983[idx2:idx3, ]
X_1992_Validate <- X_1992[idx2:idx3, ]
X_1995_Validate <- X_1995[idx2:idx3, ]
X_2003_Validate <- X_2003[idx2:idx3, ]
X_2011_Validate <- X_2011[idx2:idx3, ]
X_2020_Validate <- X_2020[idx2:idx3, ]

Y_1972_Validate <- Y_1972[idx2:idx3, ]
Y_1983_Validate <- Y_1983[idx2:idx3, ]
Y_1992_Validate <- Y_1992[idx2:idx3, ]
Y_1995_Validate <- Y_1995[idx2:idx3, ]
Y_2003_Validate <- Y_2003[idx2:idx3, ]
Y_2011_Validate <- Y_2011[idx2:idx3, ]
Y_2020_Validate <- Y_2020[idx2:idx3, ]

X_1972_Test <- X_1972[idx4:nrow(X_1972), ]
X_1983_Test <- X_1983[idx4:nrow(X_1983), ]
X_1992_Test <- X_1992[idx4:nrow(X_1992), ]
X_1995_Test <- X_1995[idx4:nrow(X_1995), ]
X_2003_Test <- X_2003[idx4:nrow(X_2003), ]
X_2011_Test <- X_2011[idx4:nrow(X_2011), ]
X_2020_Test <- X_2020[idx4:nrow(X_2020), ]

Y_1972_Test <- Y_1972[idx4:nrow(Y_1972), ]
Y_1983_Test <- Y_1983[idx4:nrow(Y_1983), ]
Y_1992_Test <- Y_1992[idx4:nrow(Y_1992), ]
Y_1995_Test <- Y_1995[idx4:nrow(Y_1995), ]
Y_2003_Test <- Y_2003[idx4:nrow(Y_2003), ]
Y_2011_Test <- Y_2011[idx4:nrow(Y_2011), ]
Y_2020_Test <- Y_2020[idx4:nrow(Y_2020), ]


# Step 6: Define Inputs
Input1972 <- layer_input(shape = c(ncol(X_1972_Train)), name = "Input1972")
Input1983 <- layer_input(shape = c(ncol(X_1983_Train)), name = "Input1983")
Input1992 <- layer_input(shape = c(ncol(X_1992_Train)), name = "Input1992")
Input1995 <- layer_input(shape = c(ncol(X_1995_Train)), name = "Input1995")
Input2003 <- layer_input(shape = c(ncol(X_2003_Train)), name = "Input2003")
Input2011 <- layer_input(shape = c(ncol(X_2011_Train)), name = "Input2011")
Input2020 <- layer_input(shape = c(ncol(X_2020_Train)), name = "Input2020")

P1 <- Input1972 %>% layer_dense(200, activation = "selu")
P2 <- Input1983 %>% layer_dense(200, activation = "selu")
P3 <- Input1992 %>% layer_dense(200, activation = "selu")
P4 <- Input1995 %>% layer_dense(200, activation = "selu")
P5 <- Input2003 %>% layer_dense(200, activation = "selu")
P6 <- Input2011 %>% layer_dense(200, activation = "selu")
P7 <- Input2020 %>% layer_dense(200, activation = "selu")


# Step 7: Define Hidden Layers
features <- layer_concatenate(list(P1, P2, P3, P4, P5, P6, P7)) %>%
  layer_dense(400, activation = "selu") %>%
  layer_dropout(0.5)


# Step 8: Define Outputs
Output1972 <- features %>% layer_dense(200, activation = "selu") %>%
  layer_dense(ncol(Y_1972_Train) , activation = "softmax", name = "Output1972")
Output1983 <- features %>% layer_dense(200, activation = "selu") %>%
  layer_dense(ncol(Y_1983_Train) , activation = "softmax", name = "Output1983")
Output1992 <- features %>% layer_dense(200, activation = "selu") %>%
  layer_dense(ncol(Y_1992_Train) , activation = "softmax", name = "Output1992")
Output1995 <- features %>% layer_dense(200, activation = "selu") %>%
  layer_dense(ncol(Y_1995_Train) , activation = "softmax", name = "Output1995")
Output2003 <- features %>% layer_dense(200, activation = "selu") %>%
  layer_dense(ncol(Y_2003_Train) , activation = "softmax", name = "Output2003")
Output2011 <- features %>% layer_dense(200, activation = "selu") %>%
  layer_dense(ncol(Y_2011_Train) , activation = "softmax", name = "Output2011")
Output2020 <- features %>% layer_dense(200, activation = "selu") %>%
  layer_dense(ncol(Y_2020_Train) , activation = "softmax", name = "Output2020")


# Step 9: Define Model
model <- keras_model(
  inputs = list(Input1972, Input1983, Input1992, Input1995, Input2003, Input2011, Input2020),
  outputs = list(Output1972, Output1983, Output1992, Output1995, Output2003, Output2011, Output2020)
)

model %>% compile(
  optimizer = "adam",
  loss = c("categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy",
           "categorical_crossentropy", "categorical_crossentropy","categorical_crossentropy",
           "categorical_crossentropy"),
  metrics = c("accuracy", "top_k_categorical_accuracy")
)


# Step 10: Fit Model and Evaluate
model %>% fit(
  x = list(X_1972_Train, X_1983_Train, X_1992_Train, X_1995_Train, X_2003_Train,
           X_2011_Train, X_2020_Train),
  y = list(Y_1972_Train, Y_1983_Train, Y_1992_Train, Y_1995_Train, Y_2003_Train,
           Y_2011_Train, Y_2020_Train),
  epochs = 3
)

model %>% evaluate(
  x = list(X_1972_Validate, X_1983_Validate, X_1992_Validate, X_1995_Validate, X_2003_Validate,
           X_2011_Validate, X_2020_Validate),
  y = list(Y_1972_Validate, Y_1983_Validate, Y_1992_Validate, Y_1995_Validate, Y_2003_Validate,
           Y_2011_Validate, Y_2020_Validate)
)


# Step 11: Calculate Test Set Accuracy
model %>% evaluate(
  x = list(X_1972_Test, X_1983_Test, X_1992_Test, X_1995_Test, X_2003_Test,
           X_2011_Test, X_2020_Test),
  y = list(Y_1972_Test, Y_1983_Test, Y_1992_Test, Y_1995_Test, Y_2003_Test,
           Y_2011_Test, Y_2020_Test)
)


# Step 12: Save Model
save_model_tf(model, "OCC_Model")
