The Jupyter notebooks in this directory contain the code for DDM analyses on data from Zhao et al. (2020).

risk_data.csv is the original dataset from this paper, which is the one I have used for DDM analyses. 

risk_data_cleaned.csv is to be used for LCA analyses. It was created by:
    for each gain, loss combination:
        collect all trials (from all participants) that have these stakes
        convert respone-times to z-scores
        remove trials with |z| > 3
I seem to have deleted the notebook which was used to perform this, but you can find similar ones for other datasets (for e.g., dataset2)

There is one notebook for each of the full and restricted DDM models. I would recommend going through the HDDM tutorial, and I think the code in these notebooks is easy to follow.

savedModels contains fitted models, so you can simply load them instead of fitting them again.

simulatedData contains data generated using these fitted models.
        
plotChoiceFitsR2.py makes an R^2 plot of acceptance rates of models and humans (each point represents one gain, loss combination).

plotEmpiricalPAcceptVsAdjustedRT.py plot P_accept vs. choice-factor adjusted RTs (as discussed in Zhao et al. as well as our paper).

dataset3 contains scripts for plot R^2 plots for response times and quantile probability functions.