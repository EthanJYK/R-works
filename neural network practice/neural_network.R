# NOTE: indexing
# 1 to l: W, b, Z, dW, dB, dZ
# 1 to l+1(0 to l): layer_dims, A -> do index - 1 in order to make it correspond
#                                   to the layer numbers

# Y must be an 1 by m matrix

#------------------------- PRELIMINARY FUNCTIONS ------------------------------#
#------------------------------------------------------------------------------#
# sigmoid function
sigmoid <- function(x) {
    1 / (1 + exp(-x))
}

# ReLU function
relu <- function(x) {
    #ifelse(x > 0, x, 0)
    x[x < 0] <- 0 # faster than ifelse()
    return(x)
}

# Relu differential function
dxrelu <- function(x) {
    #ifelse(x > 0, 1, 0)
    x[x > 0] <- 1 # faster than ifelse()
    x[x < 0] <- 0
    return(x)
}

# Sigmoid differential function
dxsigmoid <- function(x) {
    x * (1-x)
}
# Derivative of cost function
dxcost <- function(AL, Y) { 
    (1-Y) / (1-AL) - Y / AL
}
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# ex) layer_dims <- c(dim(X)[1], 10, 10, 10, 1)

# initialize parameters
initialize_parameters <- function(layer_dims) {
    # inner function
    generateW <- function(n, n_1) { 
        # matrix(rnorm(n * n_1), n, n_1) * 0.01 # doesn't work like python test
        matrix(rnorm(n * n_1) * sqrt(2/n_1), n, n_1) # He initialization
    }

    # main
    n <- layer_dims[2:length(layer_dims)] # n features of the next layer input
    n_1 <- layer_dims[1:(length(layer_dims)-1)] # n of the current input
    W <<- mapply(generateW, n, n_1)  # mapply returns in a list
    b <<- lapply(n, numeric) # create n zeros, lapply returns in a list
    #b <<- lapply(n, function(x){matrix(numeric(x))})
} # W[W1, W2, W3 ... WL], b[b1, b2, b3 ... bL]


# activation = vector of functions relu or sigmoid has length of the layers 
#                                                      = length(layer_dims) - 1
# ex) activation <- c(relu, relu, relu, relu, sigmoid) 

# forward propagation
forward_propagate <- function(X, W, b, activation){
    A <- list(X) # set X as A0, combine different matrices in a list
    Z <- list()
    for (l in 1:length(W)){
        z <- W[[l]] %*% A[[l]] + b[[l]] # linear calculation
        a <- activation[[l]](z)
        Z <- c(Z, list(z))
        A <- c(A, list(a))
    }
    assign("Z", Z, envir = .GlobalEnv) # assign to .GlobalEnv
    assign("A", A, envir = .GlobalEnv)
}

# calculate cost
compute_cost <- function(Yhat, Y) {
    m <- dim(Y)[2]
    c <- -(log(Yhat) %*% t(Y) + log(1-Yhat) %*% t(1-Y))/m
    return(c)
}
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# derivative functions of activation functions
# ex) dxfunction <- c(dxrelu, dxrelu, dxrelu, dxrelu, sigmoid)

# backpropagation
backpropagate <- function(W, b, Z, A, Y, dxfunction) {
    # get dZ using for loop
    dAL <- dxcost(A[[length(A)]], Y)
    dZL <- dAL * dxfunction[[length(dxfunction)]](A[[length(A)]])
    # dZL <- A[[length(A)]] - Y # since sigmoid, we can get dZL directly.
    dZ <- list(dZL)
    for (l in (length(W)-1):1) {
        dZl <- t(W[[l+1]]) %*% dZ[[1]] * dxfunction[[l]](Z[[l]]) 
        dZ <- c(list(dZl), dZ) # the former comes the former
    }
    m <- dim(Y)[2]
    # get dW at once, in the .GlobalEnv
    dW <<- mapply(function(x, y){x%*%t(y)/m}, dZ, A[1:length(W)], 
                  SIMPLIFY = FALSE, USE.NAMES = FALSE)
    # get db at once, in the .GlobalEnv
    db <<- lapply(dZ, function(x){rowSums(x)/m}) #rowSums gives the sum of each row 
}

update <- function(W, b, dW, db, learning_rate) {
    W <<- mapply(function(x, y){x - learning_rate * y}, W, dW, 
                 SIMPLIFY = FALSE, USE.NAMES = FALSE)
    b <<- mapply(function(x, y){x - learning_rate * y}, b, db)
}
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
predict_test <- function(X, W, b, activation){
    A <- list(X) # set X as A0, combine different matrices in a list
    Z <- list()
    for (l in 1:length(W)){
        z <- W[[l]] %*% A[[l]] + b[[l]] # linear calculation
        a <- activation[[l]](z)
        Z <- c(Z, list(z))
        A <- c(A, list(a))
    }
    assign("test_Z", Z, envir = .GlobalEnv) # assign to .GlobalEnv
    assign("test_A", A, envir = .GlobalEnv)
}
#------------------------------------------------------------------------------#
#---------------------- END of PRELIMINARY FUNCTIONS --------------------------#
