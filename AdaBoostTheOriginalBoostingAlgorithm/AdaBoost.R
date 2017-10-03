
# load packages -----------------------------------------------------------

pacman::p_load(dplyr, ggplot2, latex2exp)

# inputs and settings --------------------------------------------------------

theme_set(theme_gray(base_size = 22))

# assign hex color codes used in ggplot2
TT <- "#F8766D"
FF <- "#00BFC4"
FP <- "#FB6A4A"
TP <- "#74C476"

m <- 10^5                 # size of dataset
sd <- 0.13                # noise
T_ <- 30                  # number of boosting rounds 
H.cardinality <- 100      # cardinality of hypothesis space, assumed to be a even number

# simulate data -----------------------------------------------------------

# simulate on circle, label -1
theta <- runif(m, 0, 2*pi)
circ.x <- cos(theta) + rnorm(m,sd=sd) 
circ.y <- sin(theta) + rnorm(m,sd=sd)
circ.df <- data.frame(label = -1, x = circ.x, y = circ.y)

# simulate on center of circle, label 1
center.x <- rnorm(m,sd=sd)
center.y <- rnorm(m,sd=sd)
center.df <- data.frame(label = 1, x = center.x, y = center.y)

# bind the two, add a constant for threshold and sample the data to prepare for train/test split
df <- rbind(circ.df, center.df)
df <- cbind(df, 1)
df <- df[sample(1:dim(df)[1],dim(df)[1]),]


# construct train and test set
train <- df[1:round(dim(df)[1]/2),]
test <- df[round(dim(df)[1]/2):dim(df)[1],]
train.X <- train[,-1]
train.y <- train[,1]
test.X <- test[,-1]
test.y <- test[,1]

# weak classifiers --------------------------------------------------------

# construct classifiers 
grid <- seq(-1.5, 1.5, length.out = 50)
Hypothesis.space <- rbind(cbind(1,0,grid),
                          cbind(-1,0,grid), 
                          cbind(0,1,grid),
                          cbind(0,-1,grid))

# a function for fitting to the data
fit <- function(X, y, H.space, weights){
  
  # predictions
  preds_ <- sign(H.space %*% t(X))
  
  # initiate weigthed error
  weighted_err <- rep(0, dim(H.space)[1])
  
  # fill vector with weighted errors
  for(i in 1:dim(H.space)[1]){
    weighted_err[i] <- sum(ifelse(preds_[i,] != y,1,0) * weights)
  }
  
  # return the minimum weigthed error and the unique parametrization for the best hypothesis
  res <- list(min(weighted_err), H.space[which.min(weighted_err),])
  
  return(res)
}

# a function for predicting
predict <- function(X, H, alphas){
  
  preds_ <- rep(0,dim(X)[1])
  
  for(i in 1:length(H)){
    preds_ <- preds_ + alphas[[i]] * sign(H[[i]] %*% t(X))
  }
  preds_ <- sign(preds_)
  
  return(preds_)
}


# the algorithm -----------------------------------------------------------

# initiate lists and vectors to save elements in each boosting iteration
H <- list()
alphas <- list()

# save training- and testing-error for evaluation
train_err <- c()
test_err <- c()

# initiate weights
w <- rep(1/nrow(train.X),nrow(train.X))

# the loop for the algorithm
for(t in 1:T_){
  
  #### AdaBoost ####
  
  # fit to data
  res <- fit(train.X, train.y, Hypothesis.space, w)
  
  # weighted error
  e_t <- res[[1]]
  
  # best hypothesis
  h_t <- res[[2]]
  
  # the alpha
  alpha_t <- 1/2 * log((1-e_t)/e_t)
  
  # the normalization constant
  Z_t <-  2*sqrt(e_t*(1-e_t))
  
  # update weights
  w <- w * exp(-alpha_t*train.y*predict(train.X,list(h_t),1)) / Z_t
  
  # save alphas and hypothesis
  alphas[[t]] <- alpha_t
  H[[t]] <- h_t
  
  
  
  
  
  #### Evaluate ####
  
  # save train- and test-error
  train_err <- c(train_err, mean(predict(train.X,H,alphas) != train.y))
  test_err <- c(test_err, mean(predict(test.X,H,alphas) != test.y))
  
  # print and plot each iteration
  cat("\n\nround: ",t)
  cat("\nin-sample err: ", mean(predict(train.X,H,alphas) != train.y))
  cat("\nout-of-sample err: ", mean(predict(test.X,H,alphas) != test.y))
}

# plots -------------------------------------------------------------------

# plot of data set
data.plot <- ggplot(data = data.frame(X=train.X[,1],
                                      Y=train.X[,2],
                                      Label=ifelse(train.y==1,"Positive","Negative"))) + 
  geom_point(aes(x=X,y=Y,color=Label)) + 
  scale_color_manual(values = c("Positive" = TT, "Negative" = FF))+
  ylab(NULL) +
  xlab(NULL)
data.plot

# plot the hypothesees over the dataset
for(i in 1:H.cardinality){
  if(Hypothesis.space[i,1]==1){
    data.plot <- data.plot + 
      geom_vline(xintercept = Hypothesis.space[i,3])
  } else{
    data.plot <- data.plot + 
      geom_hline(yintercept = Hypothesis.space[i,3])
  }
}
data.plot

# plot the dependency of alpha_t on epsilon_t
eps <- seq(0,1,by=0.0001)
ggplot(data = data.frame(x=eps,y=0.5*log((1-eps)/eps)),aes(x=x,y=y))+geom_line()+
  xlab(TeX('$\\epsilon_t$'))+
  ylab(TeX('$\\alpha_t$'))

# plot of the train_err, test_err and generalization bound
m<-dim(train.X)[1]
delta=0.05
H.cardinality <- dim(Hypothesis.space)[1]
omega <- sqrt((32*(1:T_*log(exp(1)*m*H.cardinality/1:T_)+log(8/delta)))/m)
plot.complexity <- ggplot() + 
  geom_line(aes(x=c(1:T_),y=train_err), color='blue') + 
  geom_line(aes(x=c(1:T_),y=test_err), colour='red') + 
  geom_line(aes(x=c(1:T_),y=train_err + omega), colour='green') +
  xlab("T") + ylab(TeX('$\\widehat{R}(H_T)+\\Omega(H_T)$')) + 
  theme(legend.position="none")
plot.complexity
