cluster_type = "SOCK"

#------------------------------- Load Data ------------------------------------#
# use neural_network.R
library(tictoc)
library(rhdf5)
h5ls("train_catvnoncat.h5") # place files in your work directory

# train set load
train_set_x <- h5read("train_catvnoncat.h5", "train_set_x")
train_set_y <- h5read("train_catvnoncat.h5", "train_set_y")

# test set load
h5ls("test_catvnoncat.h5")
test_set_x <- h5read("test_catvnoncat.h5", "test_set_x")
test_set_y <- h5read("test_catvnoncat.h5", "test_set_y")
#------------------------------------------------------------------------------#


#------------------------------ Image Check -----------------------------------#
# [3, 64, 64, 209] -> [a, b, c, d]
# d 209 -> right order
# b, c 64, 64 -> should be transposed
# a <- right order (1, 2, 3) = (r, g, b)

a <- train_set_x[, , , 1] #take 3th image = cat, 3 * 64 * 64
a1 <- a[1, , ]; a2 <- a[2, , ]; a3 <- a[3, , ] # split into 64 * 64 matrices
b1 <- t(a1); b2 <- t(a2); b3 <- t(a3)

library(abind)
c <- abind(b1, b2, b3, along = 3) # along 3rd dimension
d <- c/255

library(grid)
grid.raster(d)

library(raster)
f <- brick(c)
plotRGB(f, r=1, g=2, b=3, scale = 255)
#------------------------------------------------------------------------------#



#------------------------------ Preprocessing ---------------------------------#
# reform X
library(doSNOW)
cores <- parallel::detectCores(logical = FALSE) 
cl <- makeCluster(cores, type = cluster_type) # create doSNOW clusters
registerDoSNOW(cl)
X <- foreach (i = 1:dim(train_set_x)[4], .combine = cbind) %dopar% {
    a <- as.vector(train_set_x[, , , i]) # the same as coursera assignment
    # a <- train_set_x[, , , i]
    # a <- c(as.vector(t(a[1, ,])), as.vector(t(a[2, ,])), as.vector(t(a[3, ,])))
    return(a)
}
stopCluster(cl)
dim(X)
X <- X/255

# reform Y
Y <- t(matrix(train_set_y))
dim(Y)

# reform test_X
library(doSNOW)
cores <- parallel::detectCores(logical = FALSE) 
cl <- makeCluster(cores, type = cluster_type) # create doSNOW clusters
registerDoSNOW(cl)
test_X <- foreach (i = 1:dim(test_set_x)[4], .combine = cbind) %dopar% {
    a <- as.vector(test_set_x[, , , i]) # the same as coursera assignment
    # a <- test_set_x[, , , i]
    # a <- c(as.vector(t(a[1, ,])), as.vector(t(a[2, ,])), as.vector(t(a[3, ,])))
    return(a)
}
stopCluster(cl)
dim(test_X)
test_X <-test_X/255

# reform test_Y
test_Y <- t(matrix(test_set_y))
dim(test_Y)
#------------------------------------------------------------------------------#



#-------------------------------- Training NN ---------------------------------#
# initialize hyper parameters
set.seed(9)
layer_dims <- c(dim(X)[1], 20, 7, 5, 1)
activation <- c(relu, relu, relu, sigmoid)
dxfunction <- c(dxrelu, dxrelu, dxrelu, dxsigmoid)
learning_rate <- 0.0075
num_iterations = 2500
cost <- vector()
test_accuracy <- vector()

# initalize parameters
initialize_parameters(layer_dims)

tic()
# gradient descent
for (i in 0:num_iterations){
    forward_propagate(X, W, b, activation)
    predict_test(test_X, W, b, activation)
    
    # print results anc check stop points
    c <- compute_cost(A[[length(A)]], Y)
    at <- sum(round(test_A[[length(test_A)]]) == test_Y) / dim(test_Y)[2]
    if (c < 0.1 & at >= 0.8) break # break if cost goes below specific value
    
    # backpropagate
    cost <- c(cost, c)
    test_accuracy <- c(test_accuracy, at)
    if (i == 0 | i %% 100 == 0){
        print(paste("Iteration:", i, " Cost:", c, " Test Accuracy:", at, sep = " "))
    }
    
    backpropagate(W, b, Z, A, Y, dxfunction)
    update(W, b, dW, db, learning_rate)
}
toc()
rm(c, i)
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# check accuracy
train_accuracy <- sum(round(A[[5]]) == Y) / dim(Y)[2]
train_accuracy

sum(round(test_A[[5]]) == test_Y) / dim(test_Y)[2]

# train test set
predict_test(test_X, W, b, activation)

test_accuracy <- sum(round(test_A[[5]]) == test_Y) / dim(test_Y)[2]
test_accuracy
#------------------------------------------------------------------------------#