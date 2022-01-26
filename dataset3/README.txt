The Jupyter notebooks in this directory contain the code for DDM analyses on data from Sheng et al. (2020)

data.csv is the original dataset from this paper. data_formatting.ipynb simply formats this data to make it similar to the structure in dataset1, and stores it in data_preprocessed.csv, which is used for DDM analyses. 

data_cleaned_250_10500_zScore3.csv is to be used for LCA analyses. It was created using data clean.ipynb:
    remove trials with RT > 10.5 seconds and RT < 250 ms (selected mostly arbitrarily)
    for each gain, loss combination:
        collect all trials (from all participants) that have these stakes
        convert respone-times to z-scores
        remove trials with |z| > 3

There is one notebook for each of the full and restricted DDM models. I would recommend going through the HDDM tutorial, and I think the code in these notebooks is easy to follow.

savedModels contains fitted models, so you can simply load them instead of fitting them again.

simulatedData contains data generated using these fitted models.
        
plotChoiceFitsR2.py makes an R^2 plot of acceptance rates of models and humans (each point represents one gain, loss combination).

plotEmpiricalPAcceptVsAdjustedRT.py plots P_accept vs. choice-factor adjusted RTs (as discussed in Zhao et al. as well as our paper).

plotMeanRTFit.py makes R^2 of response times quantiles.

plotQPF.py plots quantile probability functions.