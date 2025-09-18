## This tool can perform the following tasks

1.Load a .csv or .xls/xlsx file with time and allele data.

2.If it has distances → tick “Has Distance Column”.

3.Click “Plot Mean” to preview trends.

4.Use “Plot Paired/Unpaired” or “Plot Instantaneous”

5.Try “Plot Cross-Covariance” to explore signal correlations.

6.It also generates random unpaired datasets and plots their correlations.

7.Clear and Save plots as needed.


### To Launch run the Python script:

python tool_for extrinsic_noise.py

### Loading Data

1. If your dataset includes a distance column:
    Check the box “Has Distance Column” before loading.
    If no distance column:
    Leave the checkbox unchecked.

2. Click “Load Data”
3. Choose a CSV (.csv) or Excel (.xls / .xlsx) file.

    The file must contain at least two columns:
    First column: Time values.
    Remaining columns: Paired data (Y,X) or triplets (Y,X,D).
    
    Data will be read as (Y, X, D) triplets.
    or
    Data will be read as (Y, X) pairs and symmetrized.
