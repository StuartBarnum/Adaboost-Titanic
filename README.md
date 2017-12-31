# Adaboost-Titanic
## An implementation of the Adaboost meta-algorithm, written in R and and applied to a processed version of the Kaggle Titanic data

This project has various motivations, beginning with my interest in exploring the behavior of the Adaboost algorithm, with respect to
both its accuracy and the behavior of the quantities responsible for the boosting: the various weights applied to the examples
in the training dataset and the weights applied to predictions made by the "weak" classifiers. (On adaboost, or "adaptive boosting," 
see, for example, http://rob.schapire.net/papers/explaining-adaboost.pdf .)  The Titanic dataset is relatively small and simple, thus
allowing for relatively fast processing and cross-checking, and examimation of the behavior of the algorithm over successive iterations
of boosting.

Another motivation for this project was the fact that the Cart (classification and regresssion tree) algorithms, in contrast to other 
algorithms such as Random Forest, allow that the predictive variables have missing (e.g. NA) values. Adabost can be built on top of Cart 
(written so as to "boost" Cart). And, by writing the implementation of the boosting algorithm myself, I have been able to experiment with
certain predictive variables that (perhaps) should not simply be imputed (filled in) with a package such as Mice, and experiment with other aspects, for which it has been useful to have written the implementation myself.

One such other aspect is based on an hypothesis that, to at least some non-zero degree, families on the stricken ship tended to stand
or fall together. That is, supposing that a certain member of a family survived (and controling for factors such as passenger class), 
other family members would thereby have been at least somewhat more likely to have survived. However, testing this hypothesis 
and using an associated variable as a means of prediction present various technical difficulties associated with both overfitting and 
circularity. For example, it might seem natural to use the survival percentage, for each family represented on the ship, as a predictive
variable for the members of the a family. In this case, (for example) for each of the families that have exactly one representative on the
ship, the survival percentage is either zero or 100 percent, depending on whether that individual survives. Clearly not a useful predictor!
However, for an individual for whom we don't know the result (of survival versus non-survival), this average may be a useful predictor
(again, even if we control for factors such as passenger class). One of my aims was make such an average (or something like this
average, based on surnames) a useful predictor, and to somehow overcome the model-overfitting issues (associated with the circularity issue) that a variable such as this can
introduce.

