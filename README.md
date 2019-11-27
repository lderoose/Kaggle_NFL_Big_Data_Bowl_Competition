# Kaggle_NFL_Big_Data_Bowl_Competition

Details of the competition : https://www.kaggle.com/c/nfl-big-data-bowl-2020/overview

Evaluation : Submissions will be evaluated on the Continuous Ranked Probability Score (CRPS). For each PlayId, you must predict a cumulative probability distribution for the yardage gained or lost. In other words, each column you predict indicates the probability that the team gains <= that many yards on the play. The CRPS is computed as follows:

$C=1199N∑m=1N∑n=−9999(P(y≤n)−H(n−Ym))2$,

where P is the predicted distribution, N is the number of plays in the test set, Y is the actual yardage and H(x) is the Heaviside step function (H(x)=1
for x≥0

and zero otherwise).

The submission will not score if any of the predicted values has

P(y≤k)>P(y≤k+1)

for any k (i.e. the CDF must be non-decreasing).
