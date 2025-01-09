This file is developed for monthly data analysis. CRMS datasets should be stored in the "CRMS2Map/Input" folder

1. Monthly_analysis_practice.py: Using a subset of data (e.g., Temperature, Salinity, and Water Levels), summarize the information by basin, sub-basin, and vegetation community.
	o	Read monthly continuous and discrete hydrographic datasets.
	o	Generate 12-month moving average datasets.
	o	Examine short- (15 years) and long-term (over 40 years) trends for climate driver and CRMS data.
	o	Grouped by subdomain and vegetation datasets
	o	Generates visualizations for subdomain and vegetation-specific datasets.
	o	Analyzes correlations between subdomain/vegetation datasets and climate drivers

2. Bootstrap_Regression_analysis.py: Machine learning for stepwise linear regression and random forest regression (prerequired data from Monthly_analysis_practice.py)
	regression analysis
	o	Automated bootstrap regression analysis using ordinal linear and random forest models
	o	Evaluate the performance of models

3. Regression_analysis_plot.py: Plot stepwise linear and random forest regression analyses (prerequired data from Monthly_analysis_practice.py and Bootstrap_Regression_analysis.py)
	o	Generates a time series of visualizations for each subdomain.
	o	Generates a summary table of model performance for each subdomain

	