# SIG x UNSW Algothon 2025

This repository contains our strategy implementations for the UNSW x Susquehanna International 
Group Algothon. 

### Team Members
[John Pioc](https://www.linkedin.com/in/john-pioc/), [April Kidd](https://www.linkedin.com/in/aprilkidd/), [Kshitiz Suwal](https://www.linkedin.com/in/kshitizsuwal/), [Jack Ryan](https://www.linkedin.com/in/jack-ryan-7ab1a5294) 

### Scoring
| Round         | Placing |
|---------------|---------|
| Interim Round | 39th    |
| General Round | 63rd    |

### Overview
The Algothon is a algorithmic-trading competition where teams create Python-based trading 
algorithms to trade financial instruments inside a simulated market. 

Our final trading strategy traded on 50 financial instruments using strategies such as Donchian 
Breakout, EMA Crossovers and Trade Dependence. We also used a 125-day rolling window to 
reoptimise our trading strategy to accomodate for market regime changes.

To arrive at our final strategy, we tried and tested many approaches such as:

- Machine Learning models - Linear Regression, Logistic Regression, Markov Auto Regression and 
  ARIMA. This was dropped as the data violated the assumptions of the model or the results were 
  not statistically significant
- Pairs Trading. This was dropped as market regime and its statistical properties changed 
  drastically multiple times within the seen and unseen dataset, and retraining our model was 
  too computationally expensive with respect to the 10 minute algorithim runtime
- Geometric Brownian Motion. The drift parameter (often denoted by 'μ') in GBM represents the expected 
  average rate of return of the asset. This can be used to find overall momentum by fitting GBM over a 
  lookback. This had promising results, but was unable to be implemented robustly
  before the general round submission deadline.

Overall our team spent over 600 hours testing, analysing and implementing strategies for this competition. 
The final submission for the general round is found inside the His name is Yang folder. The other folders contain 
our notebooks, testing and other files.

### Conclusion
Our final strategy encountered some issues. The data itself held no significance at any lag for ACF, and was incredibly difficult to model.
To combat this, we implemented a rolling optimisation method which would look at previous scores for assets under certain
strategies, and reallocate (or bench) assets that were underperforming. The issue with this is if the first 125 days (until the reallocation is triggered
is all negative, every asset can be benched. This is what my assumption is for why the model failed on the data. The opimisation method benched all assets, 
resulting in only 125 days of trading, all of which were losses. The scoring for this competition was based off the formula `Mean PnL - 0.1*Std Dev`, so
if the first set of days were negative, and all other days result in 0, the score will still remain low.
