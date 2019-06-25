# Batch Normalization with Adam and Minibatch Gradient Descent

# dW, db, Znorm, Ztilde, A: reset after each iteration
#                           = initialize empty vectors/list  
# W, b, EWA of mu, theta: updated(replaced) after each iteration


#------------------------- PRELIMINARY FUNCTIONS ------------------------------#
#------------------------------------------------------------------------------#
# ReLU function
relu <- function(x) {
    #ifelse(x > 0, x, 0)
    x[x < 0] <- 0 # faster than ifelse()
    return(x)
}

# Softmax function
softmax <- function(x) {
    softmax_vector <- function(x){
        x <- exp(x - max(x))
        x <- x/sum(x)
        return(x)
    }
    apply(x, 2, softmax_vector)
}

# Sigmoid function
sigmoid <- function(x) {
    1 / (1 + exp(-x))
}

# Relu derivative function
dxrelu <- function(x) {
    #ifelse(x > 0, 1, 0)
    x[x > 0] <- 1 # faster than ifelse()
    x[x < 0] <- 0
    return(x)
}

# Softmax differential function
dxsoftmax <- function(x) {
    dxsoftmax_vector <- function(x){
        x_max <- max(x)
        x <- ifelse(x == x_max, x * (1-x), -x * x_max)
        return(x)
    }
    apply(x, 2, dxsoftmax_vector)
}

# Derivative of softmax cost function
dzcost_softmax <- function(AL, Y) {
    AL - Y
}

# Sigmoid diffrential function
dxsigmoid <- function(x) {
    x * (1-x)
}
# Derivative of sigmoid cost function
dxcost_sigmoid <- function(AL, Y) { 
    (1-Y) / (1-AL) - Y / AL
}
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# We'll use batch normalization so term b comes only for the last layer
initialize_parameters <- function(layer_dims){
    
    generate_W <- function(n, n_1) {
        matrix(rnorm(n * n_1) * sqrt(2/n_1), n, n_1) # He initialization
    }
    
    generate_dW <- function(n, n_1) {
        matrix(rep(0, n * n_1), n, n_1) # fill with zeros -> will be replaced
    }

    # main parmaeters
    n <- layer_dims[2:length(layer_dims)] # n features of the next layer input
    n_1 <- layer_dims[1:(length(layer_dims)-1)] # n of the current input
    n_h <- layer_dims[2:(length(layer_dims)-1)] # n of th hidden layers
    W <<- mapply(generate_W, n, n_1)  # mapply returns in a list
    b <<- numeric(layer_dims[length(layer_dims)]) # b only for the last layer
    
    
    # Batch Normalization parameters (for hidden layers)
    BN_gamma <<- lapply(n_h, function(x){rep(1, x)}) # lapply returns results in a list
    BN_beta <<- lapply(n_h, function(x){rep(0, x)})
    
    # initialize mu, sigma2, VdW, SdW, Vdb, Sdb
    # initialize all with zeros <- elements will be replaced with derived values 
    # EWA of BN
    mu_EWA <<- lapply(n_h, numeric)
    sigma2_EWA <<- mu_EWA # same dimensions with mu
    mu_theta <<- mu_EWA # same dimensions with mu
    sigma2_theta <<- mu_EWA # same dimensions with mu

    # VdW, SdW
    VdW <<- mapply(generate_dW, n, n_1) 
    SdW <<- VdW
    
    #Vdb, Sdb
    Vdb <<- b
    Sdb <<- b
}

# EWA
get_EWA <- function(V, theta, beta){
    V <- beta * V + (1 - beta) * theta
}

#                   ------------ 
# normalize -> get mu, sigma2, z_norm
# matrix input, 1 matrix and 2 vectors output
normalize <- function(z, epsilon = 10^-8){ 
    m <- dim(z)[2]
    mu <- rowMeans(z) # vector
    sigma2 <- rowSums((z - mu)^2)/3 # vector
    zn <- (z - mu)/sqrt(sigma2 + epsilon)
    return(list(zn, mu, sigma2))
#    Znorm <- c(Znorm, list(zn)) # will be added by layer order
#    mu_theta[layer] <- list(mu) # replace previous values
#    sigma2_theta[layer] <- list(sigma2)
#    assign("Znorm", Znorm, envir = .GlobalEnv)
#    assign("mu_theta", mu_theta, envir = .GlobalEnv)
#    assign("sigma2_theta", sigma2_theta, envir = .GlobalEnv)
}

# fix_normalized (Ztilde)
fix_normalized <- function(Znorm, gamma, beta){
    gamma * Znorm + beta # will be added by layer order 
}

# forward propagation, #mu_EWA, sigma2_EWA ... 
forward_propagate <- function(X, W, b, activation) {

    # initialization
    A <- list(X) # X becomes 1st A(A0)
    Z <- list() # new Z
    Znorm <- list() # new Znorm
    Ztilde <- list() # new Ztilde
    
    # main
    for (l in 1:length(W)){ # W is a list so length works
        if(l == length(W)){ # in the last layer
            z <- W[[l]] %*% A[[l]] + b[[l]]
            a <- activation[[l]](z)
            #---
            Z <- c(Z, list(z))
            A <- c(A, list(a))
        } else {
            z <- W[[l]] %*% A[[l]]
            znorm_list <- normalize(z) # get Znorm, mu_theta, sigma2_theta
            ztilde <- fix_normalized(znorm_list[[1]], BN_gamma[[l]], BN_beta[[l]])
            a <- activation[[l]](ztilde)
            #--- 
            Z <- c(Z, list(z))
            Znorm <- c(Znorm, znorm_list[1])
            Ztilde <- c(Ztilde, list(ztilde))
            A <- c(A, list(a))
            #---
            mu_theta[l] <- znorm_list[2]
            sigma2_theta[l] <- znorm_list[3]
        }
    }
    # save results in the global environment
    assign("Z", Z, envir = .GlobalEnv)
    assign("A", A, envir = .GlobalEnv)
    assign("Znorm", Znorm, envir = .GlobalEnv)
    assign("Ztilde", Ztilde, envir = .GlobalEnv)
    assign("mu_theta", mu_theta, envir = .GlobalEnv)
    assign("sigma2_theta", sigma2_theta, envir = .GlobalEnv)
}

# cost
compute_cost <- function(Yhat, Y){ # softmax cross entropy cost function
    c <- -(Y * log(Yhat))
    return(sum(c)/dim(Y)[2])
}

# backpropagation
backpropagate <- function(W, b, Z, Znorm, Ztilde, A, Y){
    dZL <- dzcost_softmax(A[[length(A)]], Y) 
    dZ <- list(dZL) # later get dWL, dbL with dZL
    dZtilde <- list()
    dgamma <- list()
    dbeta <- list()
    m <- dim(Y)[2] # mini-batch size
    # main
    for(l in (length(W)-1):1){
        if(l == length(W)-1){ # only in the last layer
            dZtildel <- t(W[[l+1]]) %*% dZ[[1]] * dxfunction[[l]](Ztilde[[l]]) 
            # dZ[[1]] <- filling the list in an inverse order, 1 is the newest
        } else {
            dZtildel <- t(W[[l+1]]) %*% dZtilde[[1]] * dxfunction[[l]](Ztilde[[l]])
        }
        dZnorml <- dZtildel * BN_gamma[[l]] # element-wise by row
        dgammal <- rowSums(dZtildel * Znorm[[l]]) # Do not forget summation
        dbetal <- rowSums(dZtildel) # Do not forget summation
        dZl <- BN_gamma[[l]] / sqrt((sigma2_theta[[l]]+10^-8)) / m * 
            (m * dZtildel - dbetal - dgammal * Znorm[[l]])
        
        # add gradients in the list
        dZtilde <- c(list(dZtildel), dZtilde)
        dZ <- c(list(dZl), dZ)
        dgamma <- c(list(dgammal), dgamma)
        dbeta <- c(list(dbetal), dbeta)
    }
    # get dW, db at once
    dW <<- mapply(function(x, y){x%*%t(y)/m}, dZ, A[1:length(W)], 
                  SIMPLIFY = FALSE, USE.NAMES = FALSE)
    db <<- rowSums(dZL)/m
    
    # save dgamma, dbeta
    assign("dgamma", dgamma, envir = .GlobalEnv)
    assign("dbeta", dbeta, envir = .GlobalEnv)
}

# update with ADAM
update <- function(W, b, dW, db, VdW, SdW, Vdb, Sdb, BN_gamma, BN_beta, dgamma, dbeta, learning_rate) {
    W <<- mapply(function(x, v, s){x - learning_rate * v / (sqrt(s) + 10^-8)}, W, VdW, SdW,
                 SIMPLIFY = FALSE, USE.NAMES = FALSE)
    b <<- mapply(function(x, v, s){x - learning_rate * v / (sqrt(s) + 10^-8)}, b, Vdb, Sdb)
    BN_gamma <<- mapply(function(x, y){x - learning_rate * y}, BN_gamma, dgamma, 
                 SIMPLIFY = FALSE, USE.NAMES = FALSE)
    BN_beta <<- mapply(function(x, y){x - learning_rate * y}, BN_beta, dbeta, 
                        SIMPLIFY = FALSE, USE.NAMES = FALSE)
}
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
predict_test <- function(X){
    A <- list(X) # set X as A0, combine different matrices in a list
    Z <- list()
    Znorm <- list() # new Znorm
    Ztilde <- list() # new Ztilde
    for (l in 1:length(W)){
        if(l == length(W)){
            z <- W[[l]] %*% A[[l]] + b[[l]] # linear calculation
            a <- activation[[l]](z)
            Z <- c(Z, list(z))
            A <- c(A, list(a))
        } else {
            z <- W[[l]] %*% A[[l]]
            znorm <- (z - mu_EWA[[l]])/sqrt(sigma2_EWA[[l]] + 10^-8) 
            ztilde <- fix_normalized(znorm, BN_gamma[[l]], BN_beta[[l]])
            a <- activation[[l]](ztilde)
            #--- 
            Z <- c(Z, list(z))
            Znorm <- c(Znorm, znorm)
            Ztilde <- c(Ztilde, list(ztilde))
            A <- c(A, list(a))
        }
    }
    apply(A[[length(A)]], 2, function(x){
        max <- which.max(x)
        x <- x * 0
        x[max] <- 1
        return(x)
    })
}
#------------------------------------------------------------------------------#