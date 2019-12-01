# Kaggle_NFL_Big_Data_Bowl_Competition
************************************************************************************
$Task : predict how many yards an american footaball team will gain on rushing play$
************************************************************************************
Details of the competition : https://www.kaggle.com/c/nfl-big-data-bowl-2020/overview

Evaluation : Submissions will be evaluated on the Continuous Ranked Probability Score (CRPS). For each PlayId, you must predict a cumulative probability distribution for the yardage gained or lost. 

Simple model : predict always 3 yards will gained and applied a normal smoothing on the cumulative distribution function.
Public score : 0.01461

Final model : clean data and create features for players that refers to the position of QuarterBack. 
Multiclass classification (199 classes) fitting with LightGBM.
Public score : 0.01408
