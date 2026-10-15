# Run from repository root. Original script is deliberately unchanged.
# Only rpart is required for revised models; legacy comparison needs tidyverse/stringi.
raw_features <- function(d) {
  d$Title <- sub('^[^,]*, *([^.]+\\.).*$', '\\1', d$Name)
  d$Title[d$Title %in% c('Dona.', 'Ms.')] <- NA_character_
  d$Surname <- sub(',.*$', '', d$Name)
  d$Cabin_letter <- substr(d$Cabin, 1, 1)
  d$Cabin_letter[d$Cabin_letter == ''] <- NA_character_
  d$SibSpPlusParch <- d$SibSp + d$Parch
  d
}
age_key <- function(d) paste(ifelse(is.na(d$Title), '<missing>', d$Title), d$Pclass)
prepare_fold <- function(train, test, reference = NULL) {
  stopifnot(!anyDuplicated(train$PassengerId), !anyDuplicated(test$PassengerId),
            !any(test$PassengerId %in% train$PassengerId), !anyNA(train$Survived))
  train <- raw_features(train); test <- raw_features(test)
  # Outcome is never returned to the prediction code.
  test$Survived <- NULL
  known <- is.finite(train$Age)
  if (!any(known)) stop('No observed training ages')
  means <- tapply(train$Age[known], age_key(train)[known], mean)
  class_means <- tapply(train$Age[known], train$Pclass[known], mean)
  fill <- function(d) {
    a <- unname(means[age_key(d)])
    b <- unname(class_means[as.character(d$Pclass)])
    a[!is.finite(a)] <- b[!is.finite(a)]
    a[!is.finite(a)] <- mean(train$Age[known])
    d$Age[!is.finite(d$Age)] <- a[!is.finite(d$Age)]
    counts <- table(train$Surname)
    d$sum_Surname <- as.numeric(counts[d$Surname]); d$sum_Surname[is.na(d$sum_Surname)] <- 0
    d$Age_bin <- cut(d$Age, c(-1,6,12,21,30,48,66,100))
    d$Fare_bin <- cut(d$Fare, c(-1,7,8,12,19,50,78,600))
    d
  }
  train <- fill(train); test <- fill(test)
  if (!is.null(reference)) {
    # Honest target encoding: reference labels belong to NO learner-fit row.
    stopifnot(!any(reference$PassengerId %in% c(train$PassengerId, test$PassengerId)))
    reference <- raw_features(reference)
    rates <- tapply(reference$Survived, reference$Surname, mean)
    prior <- mean(reference$Survived)
    encode <- function(d) {
      z <- unname(rates[d$Surname]); z[is.na(z)] <- prior; z
    }
    train$frac_Surname_survived <- encode(train)
    test$frac_Surname_survived <- encode(test)
  }
  for (v in c('Pclass','Sex','Title','Cabin_letter','Embarked')) {
    levels_train <- sort(unique(as.character(train[[v]][!is.na(train[[v]])])))
    train[[v]] <- factor(train[[v]], levels = levels_train)
    # Unseen categories become missing and use CART's missing-value handling.
    test[[v]] <- factor(test[[v]], levels = levels_train)
  }
  train$Survived <- factor(train$Survived, levels = c(0,1))
  list(train = train, test = test)
}
boost_predict <- function(train, test, iterations=60L, depth=2L, cp=.02, seed=1L) {
  stopifnot(iterations > 0, depth > 0, !anyNA(train$Survived))
  if (length(unique(train$Survived)) == 1L) {
    return(matrix(as.integer(as.character(train$Survived[1])), nrow(test), iterations))
  }
  set.seed(seed)
  vars <- c('Pclass','Sex','Age_bin','Fare_bin','Title','SibSpPlusParch',
            'Cabin_letter','Embarked','sum_Surname')
  if ('frac_Surname_survived' %in% names(train)) vars <- c(vars, 'frac_Surname_survived')
  form <- reformulate(vars, 'Survived')
  y <- as.integer(as.character(train$Survived)); w <- rep(1/nrow(train), nrow(train))
  score <- rep(0, nrow(test)); out <- matrix(0L, nrow(test), iterations)
  for (i in seq_len(iterations)) {
    # Preserve original uniform bootstrap + case-weighted CART design.
    rows <- sample.int(nrow(train), nrow(train), replace=TRUE)
    fit <- rpart::rpart(form, data=train[rows,], weights=w[rows], method='class',
      control=rpart::rpart.control(maxdepth=depth, cp=cp, xval=0))
    p <- as.integer(as.character(predict(fit, train, type='class')))
    q <- as.integer(as.character(predict(fit, test, type='class')))
    err <- sum(w[p != y])
    if (!is.finite(err)) stop('Nonfinite weighted error')
    # Reject weak learners at chance or worse; retain current ensemble.
    if (err < .5) {
      alpha <- log((1-max(err,1e-12))/max(err,1e-12))
      score <- score + (q-.5)*alpha
      logw <- log(w) + ifelse(p != y, alpha/2, -alpha/2)
      w <- exp(logw-max(logw)); w <- w/sum(w)
    }
    out[,i] <- as.integer(score > 0)
    if (err <= 1e-12) {
      out[,i:iterations] <- out[,i]; break
    }
  }
  out
}
# Explicitly load definitions only; never execute original parallel main program.
load_legacy <- function(path='Adaboost_Dec_27_leave_one_out.R') {
  e <- new.env(parent=globalenv())
  expr <- parse(path)
  for (x in expr) {
    if (is.call(x) && identical(x[[1]], as.name('<-')) &&
        as.character(x[[2]]) %in% c('titanic_wrangle','titanic_adaboost')) eval(x,e)
  }
  e$c_f <- .4
  e
}
