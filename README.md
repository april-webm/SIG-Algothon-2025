# SIG x UNSW Algothon 2025

This repository contains our strategy implementations for the UNSW x Susquehanna International 
Group Algothon. 

### Team Members
[John Pioc](https://www.linkedin.com/in/john-pioc/), [April Kidd](https://www.linkedin.com/in/aprilkidd/), [Kshitiz Suwal](https://www.linkedin.com/in/kshitizsuwal/)
### Scoring
| Round         | Placing |
|---------------|---------|
| Interim Round | 39th    |
| General Round | 63rd    |
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

In the end, we ended up placing 63rd out of 180 teams.
