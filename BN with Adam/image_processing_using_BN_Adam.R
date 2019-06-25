cluster_type = "SOCK" # "FORK" for Linux

#------------------------------- Load Data ------------------------------------#
library(tictoc)
library(rhdf5)

# check file contents
h5ls("train_signs.h5")

# train set load
train_set_x <- h5read("train_signs.h5", "train_set_x")
train_set_y <- h5read("train_signs.h5", "train_set_y")

# test set load
test_set_x <- h5read("test_signs.h5", "test_set_x")
test_set_y <- h5read("test_signs.h5", "test_set_y")
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
c <- abind(b1, b2, b3, along = 3) # along the 3rd dimension
rm(a, a1, a2, a3, b1, b2, b3) # remove junks

# 1st way
library(raster)
f <- brick(c)
plotRGB(f, r=1, g=2, b=3, scale = 255)

# 2nd way
d <- c/255
library(grid)
grid.raster(d)

rm(c, d, f) # remove junks
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

# reform Y (into one-hot vectors)
Y <- matrix(rep(0, length(unique(train_set_y)) * length(train_set_y)), 
            length(unique(train_set_y)), length(train_set_y))

for(i in 1:length(train_set_y)){
    Y[train_set_y[i]+1, i] <- 1
}
rm(i)

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
test_Y <- matrix(rep(0, length(unique(test_set_y)) * length(test_set_y)), 
            length(unique(test_set_y)), length(test_set_y))

for(i in 1:length(test_set_y)){
    test_Y[test_set_y[i]+1, i] <- 1
}
rm(i, cl, cores)
#------------------------------------------------------------------------------#



#--------------------------------- Train --------------------------------------#
# Hyperparameters
set.seed(0)
layer_dims <- c(dim(X)[1], 25, 12, 6)
activation <- c(relu, relu, softmax)
dxfunction <- c(dxrelu, dxrelu, dxsoftmax)
learning_rate = 0.001 # from 0.0001 -> 0.001
num_epochs =750 # from 1500 -> 700 ~ 800 (0.9), 900 (0.89) -> between (700~800)? 750 (0.9083)
minibatch_size = 32
beta1 = 0.9
beta2 = 0.999
cost <- vector()

# initalize parameters
initialize_parameters(layer_dims)

# gradient descent
tic()
for (i in 1:num_epochs){
    for(t in 1:ceiling(dim(Y)[2]/minibatch_size)){
        batch <- (1+(t-1)*minibatch_size):min(t*32, dim(Y)[2])
        
        forward_propagate(X[,batch], W, b, activation)
        
        # print results anc check stop points
        c <- compute_cost(A[[length(A)]], Y[,batch])
        
        # backpropagate
        backpropagate(W, b, Z, Znorm, Ztilde, A, Y[,batch])
        
        # get ADAM
        VdW <- mapply(function(x,y){get_EWA(x, y, beta1)}, VdW, dW,
                      SIMPLIFY = FALSE, USE.NAMES = FALSE)
        SdW <- mapply(function(x,y){get_EWA(x, y^2, beta2)}, SdW, dW,
                      SIMPLIFY = FALSE, USE.NAMES = FALSE)
        Vdb <- mapply(function(x,y){get_EWA(x, y, beta1)}, Vdb, db)
        Sdb <- mapply(function(x,y){get_EWA(x, y^2, beta2)}, Sdb, db)
        
        
        # update with ADAM
        update(W, b, dW, db, VdW, SdW, Vdb, Sdb, BN_gamma, BN_beta, dgamma, dbeta, learning_rate)
        
        # update mu, sigma2 EWA
        mu_EWA <- mapply(function(x,y){get_EWA(x, y, beta2)}, mu_EWA, mu_theta,
                         SIMPLIFY = FALSE, USE.NAMES = FALSE)
        sigma2_EWA <- mapply(function(x,y){get_EWA(x, y, beta2)}, sigma2_EWA, sigma2_theta,
                             SIMPLIFY = FALSE, USE.NAMES = FALSE)
        
        
        
    }
    # record cost at the end of each epoch 
    cost <- c(cost, c)
    # print
    print(paste("epoch:", i, " Cost:", c, sep = " "))
}
toc()
#------------------------------------------------------------------------------#


#----------------------------------- Test -------------------------------------#
# see error-metrics in details
# reverse one-hot
# confusionMatrix requires factor-type values

# train set accuracy
library(caret)
train_results <- predict_test(X)
train_results <- as.factor(apply(train_results, 2, function(x){which(x == 1)})) 
factor_Y <- as.factor(apply(Y, 2, function(x){which(x == 1)}))
levels(train_results) <- as.character(c(0:5)) # set levels 1:6 -> 0:5
levels(factor_Y) <- as.character(c(0:5))
confusionMatrix(train_results, factor_Y)

# test set accuracy
test_results <- predict_test(test_X)
test_results <- as.factor(apply(test_results, 2, function(x){which(x == 1)}))
factor_test_Y <- as.factor(apply(test_Y, 2, function(x){which(x == 1)}))
levels(test_results) <- as.character(c(0:5)) # set levels 1:6 -> 0:5
levels(factor_test_Y) <- as.character(c(0:5))
confusionMatrix(test_results, factor_test_Y)
#------------------------------------------------------------------------------#


#------------------------ See Which are different -----------------------------#
test_results <- predict_test(test_X)
difference <- colSums(abs(test_results - test_Y))
difference <- which(difference == 2) 
difference

# see
for(i in difference){
    a <- test_set_x[, , , i] #take 3th image = cat, 3 * 64 * 64
    a1 <- a[1, , ]; a2 <- a[2, , ]; a3 <- a[3, , ] # split into 64 * 64 matrices
    b1 <- t(a1); b2 <- t(a2); b3 <- t(a3)
    
    c <- abind(b1, b2, b3, along = 3) # along the 3rd dimension
    rm(a, a1, a2, a3, b1, b2, b3) # remove junks
    
    f <- brick(c)
    plotRGB(f, r=1, g=2, b=3, scale = 255)
}

test_results[,difference]
#------------------------------------------------------------------------------#

