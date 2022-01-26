The Jupyter notebooks in this directory contain the code for DDM analyses on data from Konovalov and Krajbich (2019)

study1nonadaptive.csv is the original dataset from this paper. In a given trial, the participant has to choose between an amount they would receive for sure (100% probability) and an equiprobable mixed gamble. Note that accepting a gamble is denoted by 0 (instead of 1 as in other datasets). We are only interested in trials when the sure amount is zero. These trials are contained in dataZeroSure.csv, which is used for DDM analyses. 

dataZeroSure_cleaned_250_10500_zScore3.csv is to be used for LCA analyses. It was created using data clean.ipynb:
    remove trials with RT > 10.5 seconds and RT < 250 ms (selected mostly arbitrarily)
    for each gain, loss combination:
        collect all trials (from all participants) that have these stakes
        convert respone-times to z-scores
        remove trials with |z| > 3

There is one notebook for each of the full and restricted DDM models. I would recommend going through the HDDM tutorial, and I think the code in these notebooks is easy to follow.

savedModels contains fitted models, so you can simply load them instead of fitting them again.

simulatedData contains data generated using these fitted models.
        
plotChoiceFitsR2.py makes an R^2 plot of acceptance rates of models and humans (each point represents one gain, loss combination).

plotEmpiricalPAcceptVsAdjustedRT.py plot P_accept vs. choice-factor adjusted RTs (as discussed in Zhao et al. as well as our paper).

dataset3 contains scripts for plot R^2 plots for response times and quantile probability functions.