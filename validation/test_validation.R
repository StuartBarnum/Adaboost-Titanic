source('validation/leakage_free.R')
d <- read.csv('Kaggle_train .csv',na.strings=c('','NA'),stringsAsFactors=FALSE)
tr <- d[1:700,]; te <- d[701:710,]
a <- prepare_fold(tr,te)
# Poison held-out labels: every returned predictor must remain identical.
poison <- te; poison$Survived <- 1-poison$Survived
stopifnot(identical(a,prepare_fold(tr,poison)), !'Survived' %in% names(a$test))
# Age imputation is independent of ALL labels; test covariates cannot affect fitting.
tr2 <- tr; tr2$Survived <- 1-tr2$Survived
stopifnot(identical(a$train$Age,prepare_fold(tr2,te)$train$Age))
te2 <- te; te2$Age <- 99
stopifnot(identical(a$train,prepare_fold(tr,te2)$train))
# Known exact mean and fallback expectations, no held-out age enters estimates.
s <- d[1:4,]; s$Name <- c('A, Mr. One','B, Mr. Two','C, Mr. Three','D, Mrs. Four')
s$Pclass <- c(1,1,1,2); s$Age <- c(20,40,NA,50)
u <- d[5:7,]; u$Name <- c('E, Mr. Five','F, Dr. Six','G, Mr. Seven')
u$Pclass <- c(1,2,3); u$Age <- NA_real_
z <- prepare_fold(s,u)
stopifnot(z$train$Age[3]==30,all(z$test$Age==c(30,50,110/3)))
# Honest encoder cannot use fitting labels; reference labels may influence encoding.
ref <- d[711:800,]; ref$Name <- sub('^[^,]+','Shared',ref$Name)
f <- tr; f$Name <- sub('^[^,]+','Shared',f$Name)
t <- te; t$Name <- sub('^[^,]+','Shared',t$Name)
x <- prepare_fold(f,t,ref); f$Survived <- 1-f$Survived
y <- prepare_fold(f,t,ref)
stopifnot(identical(x$train$frac_Surname_survived,y$train$frac_Surname_survived),
  all(x$test$frac_Surname_survived==mean(ref$Survived)))
stopifnot(inherits(try(prepare_fold(tr,tr[1,,drop=FALSE]),silent=TRUE),'try-error'))
# Deterministic predictions, held-out label invariance, binary finite output.
p <- boost_predict(a$train,a$test,3,seed=7)
stopifnot(identical(p,boost_predict(a$train,prepare_fold(tr,poison)$test,3,seed=7)),
          all(p %in% 0:1),identical(dim(p),c(10L,3L)))
# Degenerate one-class training must not produce NaN boosting weights.
one <- a$train; one$Survived <- factor(rep(0,nrow(one)),levels=c(0,1))
stopifnot(all(boost_predict(one,a$test,3)==0))
cat('All validation tests passed\n')
