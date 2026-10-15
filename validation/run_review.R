# Usage: Rscript validation/run_review.R [loo|grouped|smoke] [iterations] [seed]
# Legacy needs tidyverse, stringi, stringr; revised needs rpart only.
source('validation/leakage_free.R')
args <- commandArgs(TRUE)
mode <- if(length(args)) args[1] else 'loo'
iterations <- if(length(args)>1) as.integer(args[2]) else 10L
seed <- if(length(args)>2) as.integer(args[3]) else 20261015L
stopifnot(mode %in% c('loo','grouped','smoke'))
d <- read.csv('Kaggle_train .csv', stringsAsFactors=FALSE, na.strings=c('', 'NA'))
k <- read.csv('Kaggle_test.csv', stringsAsFactors=FALSE, na.strings=c('', 'NA'))
k$Survived <- NA_integer_; k <- k[,names(d)]
if (requireNamespace('tidyverse', quietly=TRUE)) suppressPackageStartupMessages(library(tidyverse))
legacy <- load_legacy()
# Optional age-only ablation, preserving all other historical transformations.
age_fixed <- legacy$titanic_wrangle
b <- paste(deparse(body(age_fixed)), collapse='\n')
b <- sub('group_by(Title, Pclass, Survived)', 'group_by(Title, Pclass)', b, fixed=TRUE)
# This ablation is still transductive: not presented as the corrected estimator.
body(age_fixed) <- parse(text=b)[[1]]
# Preserve historical vector-&& semantics on modern dplyr (unused diagnostic column).
for (name in c('titanic_wrangle')) {
  f <- legacy[[name]]; txt <- paste(deparse(body(f)), collapse='\n')
  txt <- gsub('&&', '&', txt, fixed=TRUE); body(f) <- parse(text=txt)[[1]]; legacy[[name]] <- f
}
txt <- paste(deparse(body(age_fixed)), collapse='\n'); txt <- gsub('&&','&',txt,fixed=TRUE)
body(age_fixed) <- parse(text=txt)[[1]]
if (mode == 'grouped') {
  # Connected components prevent both surname AND ticket overlap across folds.
  parent <- seq_len(nrow(d))
  root <- function(i) { while(parent[i] != i) i <- parent[i]; i }
  for (v in list(sub(',.*','',d$Name), d$Ticket)) {
    for (ix in split(seq_len(nrow(d)), v)) if(length(ix)>1)
      for(j in ix[-1]) parent[root(j)] <- root(ix[1])
  }
  groups <- vapply(seq_len(nrow(d)), root, 1L)
  set.seed(seed); ug <- sample(unique(groups)); assignments <- rep(1:5,length.out=length(ug))
  folds <- split(seq_len(nrow(d)), assignments[match(groups,ug)])
} else folds <- as.list(if(mode=='smoke') seq_len(10L) else seq_len(nrow(d)))
methods <- c('legacy','age_only_ablation','no_target','honest_target','honest_no_target')
pred <- array(NA_integer_, c(nrow(d),iterations,length(methods)),dimnames=list(NULL,NULL,methods))
errors <- list()
for (fold in seq_along(folds)) {
  ids <- folds[[fold]]; tr <- d[-ids,,drop=FALSE]; te <- d[ids,,drop=FALSE]
  fold_seed <- seed + fold
  for (method in methods) {
    result <- tryCatch({
      if(method %in% c('legacy','age_only_ablation')) {
        test_all <- rbind(te,k)
        wr <- if(method=='legacy') legacy$titanic_wrangle(tr,test_all) else age_fixed(tr,test_all)
        set.seed(fold_seed)
        z <- legacy$titanic_adaboost(wr[[1]],wr[[2]],iterations,2,.02)$predictions
        # Explicit IDs, never assume the first row identifies the validation passenger.
        order_ids <- sort(test_all$PassengerId)
        do.call(cbind,z)[match(te$PassengerId,order_ids),,drop=FALSE]
      } else {
        if(method %in% c('honest_target','honest_no_target')) {
          set.seed(fold_seed); ix <- sample.int(nrow(tr),floor(.3*nrow(tr)))
          ref <- tr[ix,]; fit <- tr[-ix,]
          wr <- prepare_fold(fit,te,if(method=='honest_target') ref else NULL)
        } else wr <- prepare_fold(tr,te)
        boost_predict(wr$train,wr$test,iterations,seed=fold_seed)
      }
    }, error=function(e) { errors[[length(errors)+1L]] <<- data.frame(fold,method,error=conditionMessage(e)); NULL })
    if(!is.null(result)) pred[ids,,method] <- result
  }
  cat('Completed fold',fold,'of',length(folds),'\n')
}
dir.create('validation/results',showWarnings=FALSE)
prefix <- paste0('validation/results/',mode,'_',seed,'_',iterations)
saveRDS(list(predictions=pred,truth=d$Survived,ids=d$PassengerId,folds=folds,
  seed=seed,iterations=iterations,mode=mode,errors=errors,session=sessionInfo()),paste0(prefix,'.rds'))
summary <- do.call(rbind,lapply(methods,function(m) do.call(rbind,lapply(seq_len(iterations),function(i) {
  ok <- !is.na(pred[,i,m]); data.frame(method=m,iteration=i,n=sum(ok),
    correct=sum(pred[ok,i,m]==d$Survived[ok]),accuracy=if(any(ok)) mean(pred[ok,i,m]==d$Survived[ok]) else NA_real_)
}))))
write.csv(summary,paste0(prefix,'_summary.csv'),row.names=FALSE)
# Paired changes only over the SAME successfully predicted passengers; no best-iteration selection.
paired <- do.call(rbind,lapply(methods[-1],function(m) do.call(rbind,lapply(seq_len(iterations),function(i) {
  ok <- !is.na(pred[,i,m]) & !is.na(pred[,i,'legacy'])
  a <- pred[ok,i,m]==d$Survived[ok]; b <- pred[ok,i,'legacy']==d$Survived[ok]
  data.frame(method=m,iteration=i,n=sum(ok),gained=sum(a & !b),lost=sum(!a & b),delta=if(any(ok)) mean(a-b) else NA_real_)
}))))
write.csv(paired,paste0(prefix,'_paired.csv'),row.names=FALSE)
if(length(errors)) write.csv(do.call(rbind,errors),paste0(prefix,'_errors.csv'),row.names=FALSE)
print(subset(summary,iteration %in% c(1,10,20,60)))
if(length(errors)) stop('Some fits failed; inspect the errors CSV. Do not report incomplete coverage as full CV accuracy.')
