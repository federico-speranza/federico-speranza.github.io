---
title: Football
layout: post
post-image: "https://raw.githubusercontent.com/thedevslot/WhatATheme/master/assets/images/SamplePost.png?token=AHMQUEPC4IFADOF5VG4QVN26Z64GG"
description: A sample post to show how the content will look and how will different
  headlines, quotes and codes will be represented.
tags:
- football
- post
- test
---


```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import *
```


```python
teams = pd.read_csv('teams.csv')
results = pd.read_csv('results.csv')
fixtures = pd.read_csv('fixtures.csv')
players = pd.read_csv('players.csv')
startingXI = pd.read_csv('startingXI.csv')
odds = pd.read_csv('odds.csv')
```

# Exploring the First Season

## _Which team won the league in the first season?_


```python
MAX_WEEK = 54
```


```python
results.loc[results['SeasonID'] == 1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SeasonID</th>
      <th>Gameweek</th>
      <th>MatchID</th>
      <th>HomeTeamID</th>
      <th>HomeScore</th>
      <th>HomeShots</th>
      <th>AwayTeamID</th>
      <th>AwayScore</th>
      <th>AwayShots</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>17</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>8</td>
      <td>8</td>
      <td>3</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>21</td>
      <td>9</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>25</td>
      <td>10</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>13</td>
      <td>11</td>
      <td>3</td>
      <td>19</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>751</th>
      <td>1</td>
      <td>54</td>
      <td>752</td>
      <td>17</td>
      <td>1</td>
      <td>15</td>
      <td>26</td>
      <td>2</td>
      <td>9</td>
    </tr>
    <tr>
      <th>752</th>
      <td>1</td>
      <td>54</td>
      <td>753</td>
      <td>18</td>
      <td>2</td>
      <td>12</td>
      <td>25</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>753</th>
      <td>1</td>
      <td>54</td>
      <td>754</td>
      <td>19</td>
      <td>3</td>
      <td>21</td>
      <td>24</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>754</th>
      <td>1</td>
      <td>54</td>
      <td>755</td>
      <td>20</td>
      <td>0</td>
      <td>8</td>
      <td>23</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>755</th>
      <td>1</td>
      <td>54</td>
      <td>756</td>
      <td>21</td>
      <td>1</td>
      <td>21</td>
      <td>22</td>
      <td>0</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
<p>756 rows × 9 columns</p>
</div>




```python
results_season1 = initialise_table(teams)
```


```python
table_gameweek54 = generate_gameweek_table(results, results_season1, gameweek=54, seasonID=1)
```


```python
table_gameweek54.sort_values(by='Points', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TeamName</th>
      <th>TeamID</th>
      <th>Gameweek</th>
      <th>Points</th>
      <th>GF</th>
      <th>GA</th>
      <th>DIFF</th>
      <th>W</th>
      <th>D</th>
      <th>L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>Miami</td>
      <td>15</td>
      <td>54</td>
      <td>138</td>
      <td>159</td>
      <td>41</td>
      <td>118</td>
      <td>44</td>
      <td>6</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cincinnati</td>
      <td>8</td>
      <td>54</td>
      <td>125</td>
      <td>130</td>
      <td>51</td>
      <td>79</td>
      <td>39</td>
      <td>8</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Baltimore</td>
      <td>4</td>
      <td>54</td>
      <td>117</td>
      <td>136</td>
      <td>41</td>
      <td>95</td>
      <td>35</td>
      <td>12</td>
      <td>7</td>
    </tr>
    <tr>
      <th>18</th>
      <td>New York S</td>
      <td>19</td>
      <td>54</td>
      <td>113</td>
      <td>108</td>
      <td>52</td>
      <td>56</td>
      <td>34</td>
      <td>11</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Boston</td>
      <td>5</td>
      <td>54</td>
      <td>106</td>
      <td>130</td>
      <td>58</td>
      <td>72</td>
      <td>31</td>
      <td>13</td>
      <td>10</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Seattle</td>
      <td>27</td>
      <td>54</td>
      <td>105</td>
      <td>118</td>
      <td>64</td>
      <td>54</td>
      <td>31</td>
      <td>12</td>
      <td>11</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chicago B</td>
      <td>6</td>
      <td>54</td>
      <td>105</td>
      <td>110</td>
      <td>56</td>
      <td>54</td>
      <td>32</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Oakland</td>
      <td>21</td>
      <td>54</td>
      <td>96</td>
      <td>98</td>
      <td>66</td>
      <td>32</td>
      <td>27</td>
      <td>15</td>
      <td>12</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Chicago H</td>
      <td>7</td>
      <td>54</td>
      <td>95</td>
      <td>97</td>
      <td>75</td>
      <td>22</td>
      <td>28</td>
      <td>11</td>
      <td>15</td>
    </tr>
    <tr>
      <th>23</th>
      <td>St. Louis</td>
      <td>24</td>
      <td>54</td>
      <td>94</td>
      <td>108</td>
      <td>62</td>
      <td>46</td>
      <td>27</td>
      <td>13</td>
      <td>14</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Philadelphia</td>
      <td>22</td>
      <td>54</td>
      <td>82</td>
      <td>76</td>
      <td>64</td>
      <td>12</td>
      <td>24</td>
      <td>10</td>
      <td>20</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Pittsburgh</td>
      <td>23</td>
      <td>54</td>
      <td>80</td>
      <td>86</td>
      <td>89</td>
      <td>-3</td>
      <td>24</td>
      <td>8</td>
      <td>22</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Arlington</td>
      <td>1</td>
      <td>54</td>
      <td>77</td>
      <td>80</td>
      <td>60</td>
      <td>20</td>
      <td>21</td>
      <td>14</td>
      <td>19</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Minneapolis</td>
      <td>17</td>
      <td>54</td>
      <td>76</td>
      <td>89</td>
      <td>71</td>
      <td>18</td>
      <td>22</td>
      <td>10</td>
      <td>22</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Detroit</td>
      <td>11</td>
      <td>54</td>
      <td>76</td>
      <td>79</td>
      <td>75</td>
      <td>4</td>
      <td>21</td>
      <td>13</td>
      <td>20</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Toronto</td>
      <td>28</td>
      <td>54</td>
      <td>72</td>
      <td>70</td>
      <td>73</td>
      <td>-3</td>
      <td>19</td>
      <td>15</td>
      <td>20</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Kansas City</td>
      <td>13</td>
      <td>54</td>
      <td>70</td>
      <td>77</td>
      <td>76</td>
      <td>1</td>
      <td>18</td>
      <td>16</td>
      <td>20</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Montreal</td>
      <td>18</td>
      <td>54</td>
      <td>66</td>
      <td>63</td>
      <td>94</td>
      <td>-31</td>
      <td>19</td>
      <td>9</td>
      <td>26</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Los Angeles</td>
      <td>14</td>
      <td>54</td>
      <td>63</td>
      <td>63</td>
      <td>93</td>
      <td>-30</td>
      <td>17</td>
      <td>12</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Atlanta</td>
      <td>3</td>
      <td>54</td>
      <td>56</td>
      <td>68</td>
      <td>106</td>
      <td>-38</td>
      <td>13</td>
      <td>17</td>
      <td>24</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Houston</td>
      <td>12</td>
      <td>54</td>
      <td>54</td>
      <td>63</td>
      <td>110</td>
      <td>-47</td>
      <td>14</td>
      <td>12</td>
      <td>28</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cleveland Queens</td>
      <td>9</td>
      <td>54</td>
      <td>52</td>
      <td>51</td>
      <td>91</td>
      <td>-40</td>
      <td>14</td>
      <td>10</td>
      <td>30</td>
    </tr>
    <tr>
      <th>25</th>
      <td>San Francisco</td>
      <td>26</td>
      <td>54</td>
      <td>37</td>
      <td>47</td>
      <td>114</td>
      <td>-67</td>
      <td>8</td>
      <td>13</td>
      <td>33</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Denver</td>
      <td>10</td>
      <td>54</td>
      <td>35</td>
      <td>43</td>
      <td>123</td>
      <td>-80</td>
      <td>9</td>
      <td>8</td>
      <td>37</td>
    </tr>
    <tr>
      <th>24</th>
      <td>San Diego</td>
      <td>25</td>
      <td>54</td>
      <td>34</td>
      <td>41</td>
      <td>133</td>
      <td>-92</td>
      <td>8</td>
      <td>10</td>
      <td>36</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Anaheim</td>
      <td>2</td>
      <td>54</td>
      <td>32</td>
      <td>43</td>
      <td>130</td>
      <td>-87</td>
      <td>7</td>
      <td>11</td>
      <td>36</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Milwaukee</td>
      <td>16</td>
      <td>54</td>
      <td>32</td>
      <td>53</td>
      <td>120</td>
      <td>-67</td>
      <td>6</td>
      <td>14</td>
      <td>34</td>
    </tr>
    <tr>
      <th>19</th>
      <td>New York C</td>
      <td>20</td>
      <td>54</td>
      <td>21</td>
      <td>47</td>
      <td>145</td>
      <td>-98</td>
      <td>5</td>
      <td>6</td>
      <td>43</td>
    </tr>
  </tbody>
</table>
</div>



### Miami wins the first season

-----

## _At what point in the season did that team secure their league title?_


```python
gameweek_secure = None

for w in range(1, MAX_WEEK+1):
    table_gameweek = generate_gameweek_table(results, results_season1, gameweek=w, seasonID=1)
    gameweek_secure = secure_season(table_gameweek)
    
    if gameweek_secure is not None:
        break
```


```python
print(f'Miami secure the league at the {gameweek_secure}th gameweek')
```

    Miami secure the league at the 50th gameweek


### Miami secure the league at the 50th gameweek

### The way I am calculating when Miami (winner of the season as the previous question) secure the season, is based on the comparison between the first and the second team's points in the standing after every gameweek. At each iteration within the "secure_season" method, I assume all losses for the first team in the standing and all wins for the second. If the gap in points is larger than 0, it means that the first team at that gameweek secures the seaon.

### A more comprehensive check could be done also comparing head-to-head mathes in the case two teams have the same exact points at the end of a gameweek. If that check is tied,  a further one can be done on the goal differences. The team with higher goal differences secures the season.

### In this mock dataset, the above-two scenarios do not occur so I decided to implement the method to check just the difference in points.

---
---

## _What result was the biggest upset?_

### I decide the biggest upset in terms of result is the less likely to happen. At this stage, the information we have on probabilities are the odds provided by the bookmaker.

### Although the odds are not the probability of an outcome to occur, we can leverage that:
### $P(\text{outcome}) \propto \frac{1}{\text{Odds}}$ 

### I am going through the results of the season and getting the odd relative to the outcome. Then, I select the match with the highest odd.


```python
def get_actual_odds(row):
    if row['Outcome'] == 1:
        return row['Home']
    elif row['Outcome'] == -1:
        return row['Away']
    else:
        return row['Draw']
```


```python
df_season1 = results.loc[(results['SeasonID']== 1)].copy()
df_season1['Outcome'] = df_season1.apply(determine_outcome, axis=1)

merged = pd.merge(odds, df_season1, on='MatchID')
```


```python
merged['ActualOdds'] = merged.apply(get_actual_odds, axis=1)
```


```python
merged.loc[merged['ActualOdds'].idxmax()]
```




    MatchID       168.00
    Home            1.03
    Draw           23.35
    Away           75.22
    SeasonID        1.00
    Gameweek       12.00
    HomeTeamID     15.00
    HomeScore       3.00
    HomeShots      38.00
    AwayTeamID     16.00
    AwayScore       3.00
    AwayShots      10.00
    Outcome         0.00
    ActualOdds     23.35
    Name: 167, dtype: float64




```python
odds.loc[odds['MatchID'] == 168]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MatchID</th>
      <th>Home</th>
      <th>Draw</th>
      <th>Away</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>167</th>
      <td>168</td>
      <td>1.03</td>
      <td>23.35</td>
      <td>75.22</td>
    </tr>
  </tbody>
</table>
</div>



### The result which was the biggest upset is a match between Miami (Home team) and Minneapolis (Away team) where the outcome was a Draw

---
___

# Predicting the second season


```python
team15_home = results.loc[(results['SeasonID']==1) & (results['HomeTeamID']==15)]['HomeScore'].values
```


```python
team15_away = results.loc[(results['SeasonID']==1) & (results['AwayTeamID']==15)]['AwayScore'].values
```


```python
plt.hist(np.concatenate((team15_home, team15_away)), bins=20)
plt.xlabel('Goals')
plt.ylabel('Counts')
plt.show()
```

![png](/Users/federico/Library/CloudStorage/OneDrive-UniversityCollegeLondon/PhD/CV/federico-speranza.github.io/assets/images/post_football/output_28_0.png)
    


### The plot above shows an example of the distribution of goals scored by Miami throughout the season. The distribution resambles a Poisson distribution with mean $\lambda$ somewhere between 2 and 3.

### This is in line with studies I could find in the literature about modelling the outcomes of matches during a football season.

### For this reason, I decided to start off implementing a model based on the assumption that the goals scored by each team during a match follow a Poisson distribution, and they are independent.


### Along the line of the Dixon-Coles' model, I modelled the likelihood for the home and away team goals as following:

### $P(x; \lambda)=\frac{\lambda^x e^{-\lambda}}{x!}$

### where $x$ is the goals scored in the match, and $\lambda = \alpha_h \beta_a \gamma$
### (with index h: home team, a: away team)
### with $\alpha$: attack strenght, $\beta$ : defence strenght, $\gamma$: home factor

### To ensure that lambda is always positive, I fit for:
### $\log(\lambda) = \log(\alpha_h) + \log(\beta_a) + \log(\gamma)$ $\rightarrow$ $\lambda = e^{\alpha_h + \beta_a + \gamma}$
### where I re-defined, for writing semplicity, $\log(\alpha) = \alpha$, $\log(\beta) = \beta$, $\log(\gamma) = \gamma$,

### so I am actually fitting for the log of the parameters.

### The total likelihood would result in:
### $\mathcal{L}(x,y | \mathbf{\theta})=\prod_{i=1}^N \tau(\rho)\frac{e^{x_i\left(\alpha_{h_i}+\beta_{a, i}+\gamma\right)} e^{-e^{\alpha_{n_i}+\beta_{a i}+\gamma}}}{x_{i}!} \frac{e^{y_i\left(\alpha_{a, i}+\beta_{h, i}\right)} e^{-e^{\alpha_{a,}+\beta_{h, i}}}}{y!}$

### where $N$ is the number of matches in season 1, and $\tau(\rho)$ is a function which takes into account correlations between results $0-0$, $1-0$, $0-1$, $1-1$.

### The log-likelihood is:
### $ \log \mathcal{L}=\sum_{i=1}^N \log(\tau) + x_i\left(\alpha_{h, i}+\beta_{a_i}+\gamma\right)-e^{\left(\alpha_{h_i}+\beta_{a, i}+\gamma\right)} +y_i\left(\alpha_{a_i,}+\beta_{h,i}\right)-e^{\alpha_{a_i}+\beta_{h i}}-\log \left(x_i!\right) \left.-\log \left(y_{i!}\right)\right)$

### I then sample the posterior distribution of the parameters using MCMC sampler based on the Bayes Theorem:
### $P(\theta \mid x, y) \propto \mathcal{L}(x, y \mid \theta) \cdot \pi(\theta)$


```python
import emcee
from multiprocessing import Pool, cpu_count

from emcee_func import *


alpha, beta = generate_initial_alpha(table_gameweek54)
gamma = np.full(28,.2)
rho = np.full(28,-.2)
```


```python
alpha, np.sum(np.exp(alpha))
```




    (array([-0.04067913, -0.66150565, -0.20319806,  0.48994912,  0.44482869,
             0.2777746 ,  0.15200522,  0.44482869, -0.49088013, -0.66150565,
            -0.05325791, -0.27957104, -0.07890034, -0.27957104,  0.64619844,
            -0.45241385,  0.06593061, -0.27957104,  0.25942547, -0.57255816,
             0.16226172, -0.09197242,  0.03164153,  0.25942547, -0.7091337 ,
            -0.57255816,  0.34797886, -0.17421052]),
     28.0)




```python
theta = np.concatenate([alpha[:27], beta, [gamma[0]], [rho[0]]])
pos = theta + 1e-4 * np.random.randn(300, 57)
nwalkers, ndim = pos.shape

filename = "mcmc_run.h5"

#if you want to run the parallelised MCMC just set run = True
# estimated time ~ 5 mins on 4 CPUs
run = False
if run:
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    with Pool(processes=int(cpu_count()/3)) as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, args=(results,), backend=backend, pool=pool)
        sampler.run_mcmc(pos, 200, progress=True, skip_initial_state_check=True)
```


```python
plt.rcParams['text.usetex'] = False

reader = emcee.backends.HDFBackend(filename)


samples = reader.get_chain()
alpha_labels = [fr'$\alpha_{i+1}$' for i in range(27)]
beta_labels = [fr'$\beta_{i+1}$' for i in range(28)]
gamma_label = [r'$\gamma$']
rho_label = [r'$\rho$']


labels = alpha_labels + beta_labels + gamma_label + rho_label

chain_plot = True
if chain_plot:
    fig, axes = plt.subplots(ndim, figsize=(10, 50), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
```

![png](/Users/federico/Library/CloudStorage/OneDrive-UniversityCollegeLondon/PhD/CV/federico-speranza.github.io/assets/images/post_football/output_33_0.png)
    



```python
import corner

flat_samples = reader.get_chain(discard=100, flat=True, thin=15)

corner_plot=True
if corner_plot:
    
    plt.figure(figsize=(100,100))
    corner.corner(
        flat_samples, labels=labels
    );
```


    <Figure size 10000x10000 with 0 Axes>

![png](/Users/federico/Library/CloudStorage/OneDrive-UniversityCollegeLondon/PhD/CV/federico-speranza.github.io/assets/images/post_football/output_34_1.png)
    



```python
mcmc_params = []
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    mcmc_params.append(mcmc)
```


```python
teams_params = teams.copy()
```


```python
alpha_last = np.log(28-np.sum(np.exp(np.array(mcmc_params)[:,1][:27])))
```


```python
alpha_fit = np.append(np.array(mcmc_params)[:,1][:27], alpha_last)
```


```python
# double-check on the constraint of the alpha parameters
alpha_fit, np.exp(alpha_fit).sum()
```




    (array([ 0.01157139, -0.70069541, -0.22605872,  0.43342521,  0.42870471,
             0.28406565,  0.13479957,  0.42948154, -0.44563419, -0.66020374,
            -0.08065429, -0.26276242, -0.06613465, -0.28443025,  0.60082144,
            -0.44787918,  0.02732347, -0.26825249,  0.28126603, -0.50877606,
             0.07814694, -0.06494013,  0.08062334,  0.26246163, -0.66441535,
            -0.61218524,  0.32294525,  0.04768519]),
     28.0)




```python
teams_params['alpha'] = alpha_fit
teams_params['beta'] = np.array(mcmc_params)[:,1][27:55]
gamma_fit = mcmc_params[-2][1]
rho_fit = mcmc_params[-1][1]
```


```python
teams_params
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TeamName</th>
      <th>TeamID</th>
      <th>alpha</th>
      <th>beta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Arlington</td>
      <td>1</td>
      <td>0.011571</td>
      <td>-0.256435</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Anaheim</td>
      <td>2</td>
      <td>-0.700695</td>
      <td>0.480804</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Atlanta</td>
      <td>3</td>
      <td>-0.226059</td>
      <td>0.251540</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Baltimore</td>
      <td>4</td>
      <td>0.433425</td>
      <td>-0.659432</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Boston</td>
      <td>5</td>
      <td>0.428705</td>
      <td>-0.321781</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chicago B</td>
      <td>6</td>
      <td>0.284066</td>
      <td>-0.359794</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Chicago H</td>
      <td>7</td>
      <td>0.134800</td>
      <td>-0.126519</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cincinnati</td>
      <td>8</td>
      <td>0.429482</td>
      <td>-0.511153</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cleveland Queens</td>
      <td>9</td>
      <td>-0.445634</td>
      <td>0.082527</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Denver</td>
      <td>10</td>
      <td>-0.660204</td>
      <td>0.454534</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Detroit</td>
      <td>11</td>
      <td>-0.080654</td>
      <td>-0.031592</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Houston</td>
      <td>12</td>
      <td>-0.262762</td>
      <td>0.347867</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Kansas City</td>
      <td>13</td>
      <td>-0.066135</td>
      <td>-0.099961</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Los Angeles</td>
      <td>14</td>
      <td>-0.284430</td>
      <td>0.201080</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Miami</td>
      <td>15</td>
      <td>0.600821</td>
      <td>-0.705102</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Milwaukee</td>
      <td>16</td>
      <td>-0.447879</td>
      <td>0.358513</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Minneapolis</td>
      <td>17</td>
      <td>0.027323</td>
      <td>-0.103217</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Montreal</td>
      <td>18</td>
      <td>-0.268252</td>
      <td>0.181667</td>
    </tr>
    <tr>
      <th>18</th>
      <td>New York S</td>
      <td>19</td>
      <td>0.281266</td>
      <td>-0.476706</td>
    </tr>
    <tr>
      <th>19</th>
      <td>New York C</td>
      <td>20</td>
      <td>-0.508776</td>
      <td>0.611855</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Oakland</td>
      <td>21</td>
      <td>0.078147</td>
      <td>-0.237147</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Philadelphia</td>
      <td>22</td>
      <td>-0.064940</td>
      <td>-0.274786</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Pittsburgh</td>
      <td>23</td>
      <td>0.080623</td>
      <td>0.019089</td>
    </tr>
    <tr>
      <th>23</th>
      <td>St. Louis</td>
      <td>24</td>
      <td>0.262462</td>
      <td>-0.238121</td>
    </tr>
    <tr>
      <th>24</th>
      <td>San Diego</td>
      <td>25</td>
      <td>-0.664415</td>
      <td>0.501774</td>
    </tr>
    <tr>
      <th>25</th>
      <td>San Francisco</td>
      <td>26</td>
      <td>-0.612185</td>
      <td>0.338035</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Seattle</td>
      <td>27</td>
      <td>0.322945</td>
      <td>-0.228020</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Toronto</td>
      <td>28</td>
      <td>0.047685</td>
      <td>-0.143611</td>
    </tr>
  </tbody>
</table>
</div>




```python
from IPython.display import display, Math

display(Math(r'\gamma_{\text{fit}} = ' + f"{gamma_fit:.2f}"))
display(Math(r'\rho_{\text{fit}} = ' + f"{rho_fit:.2f}"))
```


$\displaystyle \gamma_{\text{fit}} = 0.37$



$\displaystyle \rho_{\text{fit}} = -0.17$


------
------

# Simulate season 2


```python
import importlib
import simulation_func
from simulation_func import *
importlib.reload(simulation_func)
```




    <module 'simulation_func' from '/Users/federico/Library/CloudStorage/OneDrive-UniversityCollegeLondon/PhD/CV/Interviews/Pythia/simulation_func.py'>




```python
final, cumulatives_df, position_counts = simulate_n_season2(teams_params=teams_params, gamma=gamma_fit,
                   fixtures=fixtures, teams=teams, n_seasons=50)
```

    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [26:14<00:00, 31.49s/it]



```python
final
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TeamName</th>
      <th>TeamID</th>
      <th>Gameweek</th>
      <th>Points</th>
      <th>GF</th>
      <th>GA</th>
      <th>DIFF</th>
      <th>W</th>
      <th>D</th>
      <th>L</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>Miami</td>
      <td>15</td>
      <td>54</td>
      <td>126</td>
      <td>125</td>
      <td>29</td>
      <td>96</td>
      <td>38</td>
      <td>9</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Baltimore</td>
      <td>4</td>
      <td>54</td>
      <td>117</td>
      <td>107</td>
      <td>34</td>
      <td>73</td>
      <td>35</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cincinnati</td>
      <td>8</td>
      <td>54</td>
      <td>111</td>
      <td>103</td>
      <td>40</td>
      <td>63</td>
      <td>33</td>
      <td>10</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Boston</td>
      <td>5</td>
      <td>54</td>
      <td>108</td>
      <td>105</td>
      <td>46</td>
      <td>58</td>
      <td>32</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>19</th>
      <td>New York S</td>
      <td>19</td>
      <td>54</td>
      <td>106</td>
      <td>90</td>
      <td>40</td>
      <td>49</td>
      <td>31</td>
      <td>11</td>
      <td>10</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chicago B</td>
      <td>6</td>
      <td>54</td>
      <td>101</td>
      <td>91</td>
      <td>45</td>
      <td>45</td>
      <td>29</td>
      <td>11</td>
      <td>12</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Seattle</td>
      <td>27</td>
      <td>54</td>
      <td>99</td>
      <td>96</td>
      <td>51</td>
      <td>45</td>
      <td>29</td>
      <td>11</td>
      <td>13</td>
    </tr>
    <tr>
      <th>26</th>
      <td>St. Louis</td>
      <td>24</td>
      <td>54</td>
      <td>96</td>
      <td>88</td>
      <td>50</td>
      <td>37</td>
      <td>28</td>
      <td>12</td>
      <td>13</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Chicago H</td>
      <td>7</td>
      <td>54</td>
      <td>88</td>
      <td>78</td>
      <td>57</td>
      <td>20</td>
      <td>25</td>
      <td>11</td>
      <td>16</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Oakland</td>
      <td>21</td>
      <td>54</td>
      <td>87</td>
      <td>73</td>
      <td>52</td>
      <td>20</td>
      <td>24</td>
      <td>13</td>
      <td>15</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Philadelphia</td>
      <td>22</td>
      <td>54</td>
      <td>85</td>
      <td>65</td>
      <td>50</td>
      <td>15</td>
      <td>23</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Arlington</td>
      <td>1</td>
      <td>54</td>
      <td>84</td>
      <td>66</td>
      <td>50</td>
      <td>16</td>
      <td>23</td>
      <td>14</td>
      <td>16</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Toronto</td>
      <td>28</td>
      <td>54</td>
      <td>82</td>
      <td>71</td>
      <td>56</td>
      <td>15</td>
      <td>22</td>
      <td>13</td>
      <td>17</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Minneapolis</td>
      <td>17</td>
      <td>54</td>
      <td>79</td>
      <td>70</td>
      <td>60</td>
      <td>9</td>
      <td>22</td>
      <td>13</td>
      <td>18</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Pittsburgh</td>
      <td>23</td>
      <td>54</td>
      <td>78</td>
      <td>73</td>
      <td>67</td>
      <td>6</td>
      <td>21</td>
      <td>13</td>
      <td>18</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Kansas City</td>
      <td>13</td>
      <td>54</td>
      <td>75</td>
      <td>64</td>
      <td>61</td>
      <td>3</td>
      <td>20</td>
      <td>13</td>
      <td>19</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Detroit</td>
      <td>11</td>
      <td>54</td>
      <td>74</td>
      <td>63</td>
      <td>63</td>
      <td>0</td>
      <td>20</td>
      <td>14</td>
      <td>19</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Montreal</td>
      <td>18</td>
      <td>54</td>
      <td>56</td>
      <td>50</td>
      <td>79</td>
      <td>-29</td>
      <td>14</td>
      <td>12</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Atlanta</td>
      <td>3</td>
      <td>54</td>
      <td>56</td>
      <td>55</td>
      <td>86</td>
      <td>-30</td>
      <td>14</td>
      <td>11</td>
      <td>27</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Los Angeles</td>
      <td>14</td>
      <td>54</td>
      <td>55</td>
      <td>50</td>
      <td>80</td>
      <td>-30</td>
      <td>14</td>
      <td>13</td>
      <td>26</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cleveland Queens</td>
      <td>9</td>
      <td>54</td>
      <td>54</td>
      <td>42</td>
      <td>71</td>
      <td>-29</td>
      <td>13</td>
      <td>14</td>
      <td>26</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Houston</td>
      <td>12</td>
      <td>54</td>
      <td>52</td>
      <td>52</td>
      <td>91</td>
      <td>-39</td>
      <td>13</td>
      <td>11</td>
      <td>28</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Milwaukee</td>
      <td>16</td>
      <td>54</td>
      <td>43</td>
      <td>43</td>
      <td>98</td>
      <td>-55</td>
      <td>10</td>
      <td>11</td>
      <td>31</td>
    </tr>
    <tr>
      <th>24</th>
      <td>San Francisco</td>
      <td>26</td>
      <td>54</td>
      <td>39</td>
      <td>36</td>
      <td>95</td>
      <td>-58</td>
      <td>9</td>
      <td>12</td>
      <td>32</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Denver</td>
      <td>10</td>
      <td>54</td>
      <td>35</td>
      <td>33</td>
      <td>103</td>
      <td>-70</td>
      <td>8</td>
      <td>10</td>
      <td>35</td>
    </tr>
    <tr>
      <th>23</th>
      <td>San Diego</td>
      <td>25</td>
      <td>54</td>
      <td>33</td>
      <td>33</td>
      <td>109</td>
      <td>-75</td>
      <td>7</td>
      <td>9</td>
      <td>36</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Anaheim</td>
      <td>2</td>
      <td>54</td>
      <td>33</td>
      <td>33</td>
      <td>109</td>
      <td>-76</td>
      <td>7</td>
      <td>10</td>
      <td>35</td>
    </tr>
    <tr>
      <th>18</th>
      <td>New York C</td>
      <td>20</td>
      <td>54</td>
      <td>32</td>
      <td>39</td>
      <td>122</td>
      <td>-83</td>
      <td>7</td>
      <td>9</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>



### The above table shows the final standing of the simulated season, over just 50 simulations as it is computationally time consuming. Ideally, I would run at least 1000 simulations. To simulate the seasons, I use the mean of the parameters extrapolated by the posterior distributions sampled.


```python
actual_results_season2 = results.loc[results['SeasonID'] == 2]
actual_cumulative_points2 = cumulative_metric(actual_results_season2, initialise_table(teams), gameweek=54, seasonID=2, metric='Points')
```


```python
standing_season2 = generate_gameweek_table(actual_results_season2, initialise_table(teams), gameweek=54, seasonID=2)

standing_season2_sorted = standing_season2.sort_values(by='Points', ascending=False)

standing_season2_sorted['Position'] = np.array(range(1,29))

final['Position'] = np.array(range(1,29))

def position_to_ordinal(position):
    suffix = ['th', 'st', 'nd', 'rd'] + ['th'] * 17
    if 10 <= position % 100 <= 20:
        suffix_position = 'th'
    else:
        suffix_position = suffix[position % 10]
    return f"{position}{suffix_position}"

final['Ordinal_Position'] = final['Position'].apply(position_to_ordinal)
standing_season2_sorted['Ordinal_Position'] = standing_season2_sorted['Position'].apply(position_to_ordinal)

merged_df = pd.merge(final[['TeamName', 'Ordinal_Position']], 
                     standing_season2_sorted[['TeamName', 'Ordinal_Position']], 
                     on='TeamName', 
                     suffixes=('_simulated', '_actual'))


merged_df.columns = ['TeamName', 'Simulated Position', 'Actual Position']

merged_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TeamName</th>
      <th>Simulated Position</th>
      <th>Actual Position</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Miami</td>
      <td>1st</td>
      <td>1st</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Baltimore</td>
      <td>2nd</td>
      <td>3rd</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cincinnati</td>
      <td>3rd</td>
      <td>5th</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Boston</td>
      <td>4th</td>
      <td>4th</td>
    </tr>
    <tr>
      <th>4</th>
      <td>New York S</td>
      <td>5th</td>
      <td>2nd</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chicago B</td>
      <td>6th</td>
      <td>7th</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Seattle</td>
      <td>7th</td>
      <td>6th</td>
    </tr>
    <tr>
      <th>7</th>
      <td>St. Louis</td>
      <td>8th</td>
      <td>8th</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Chicago H</td>
      <td>9th</td>
      <td>10th</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Oakland</td>
      <td>10th</td>
      <td>13th</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Philadelphia</td>
      <td>11th</td>
      <td>11th</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Arlington</td>
      <td>12th</td>
      <td>9th</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Toronto</td>
      <td>13th</td>
      <td>14th</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Minneapolis</td>
      <td>14th</td>
      <td>12th</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Pittsburgh</td>
      <td>15th</td>
      <td>18th</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Kansas City</td>
      <td>16th</td>
      <td>15th</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Detroit</td>
      <td>17th</td>
      <td>16th</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Montreal</td>
      <td>18th</td>
      <td>20th</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Atlanta</td>
      <td>19th</td>
      <td>25th</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Los Angeles</td>
      <td>20th</td>
      <td>23rd</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Cleveland Queens</td>
      <td>21st</td>
      <td>17th</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Houston</td>
      <td>22nd</td>
      <td>21st</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Milwaukee</td>
      <td>23rd</td>
      <td>19th</td>
    </tr>
    <tr>
      <th>23</th>
      <td>San Francisco</td>
      <td>24th</td>
      <td>26th</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Denver</td>
      <td>25th</td>
      <td>27th</td>
    </tr>
    <tr>
      <th>25</th>
      <td>San Diego</td>
      <td>26th</td>
      <td>24th</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Anaheim</td>
      <td>27th</td>
      <td>22nd</td>
    </tr>
    <tr>
      <th>27</th>
      <td>New York C</td>
      <td>28th</td>
      <td>28th</td>
    </tr>
  </tbody>
</table>
</div>



### The above table shows the simulated position and the actual position for each team. Looking at the top 5 teams, the simulated position (5th) for the New York S team is the most far from the actual position (2nd) in the final standing. New York S tends to win with a big difference in goals against weaker teams but not against middle to top teams, scoring 1 to 2 goals. That could be why the parameters of the fit ($\alpha = 0.2$, $\beta = -0.4$) show not a strong attack strenght and so the simulations penalise the team.


```python
plt.figure(figsize=(15,30))

for k, i in zip(final['TeamID'], range(1,29)):
    plt.subplot(7,4,i)
    plt.title(f'{final.loc[final['TeamID']==k, 'TeamName'].iloc[0]}')
    
    gw = cumulatives_df.loc[cumulatives_df['team_id'] == k, ['gameweek', 'cumulative_points']].gameweek.values
    cp = cumulatives_df.loc[cumulatives_df['team_id'] == k, ['gameweek', 'cumulative_points']].cumulative_points.values
    
    plt.plot(gw, cp, color='red', label='Model')
    plt.plot(list(range(0,55)), actual_cumulative_points2[k], color = 'black', label='Observed')
    plt.xlabel('Gameweek')
    plt.ylabel('Points')
    plt.ylim(0, 150)
    plt.legend(loc='upper left')
    
plt.tight_layout()
    
```

![png](/Users/federico/Library/CloudStorage/OneDrive-UniversityCollegeLondon/PhD/CV/federico-speranza.github.io/assets/images/post_football/output_52_0.png)
    


### The above plot shows the comparison between the cumulative points thoughout the season. The red line shows the posterior predictive validation of the model while the black line shows the actual cumulative points for the second season.

### Overall, the model tends to slighlty underestimate the top teams, approximate more confidently the middle teams and underestimate the bottom teams.
### As also mentioned above, New York S is significantly underestimated by the model.

### To improve the model, we can make it a hierarchical bayesian model. This approach will allow to include additional layers of structure, leading to more accurate predictions. We can introduce hyperparameters which control the distributions of the team-specific parameters, as well as hyper-priors. 

### Moreover, including additional information i.e. shots, startingXI and position of the players could make the model more accurate in its predictions. I started to explore the relation between goals and shots, noticing a correlation between the two variable. However, the data show a considerable scatter which makes hard in the first place to model the dependence of goals on shots.
### I would try the following approach:
### Assuming that $shots \sim \mathcal{N}(\mu, \Sigma^2)$ and $\mathrm{Shots} = m \cdot \mathrm{Goals} + q$
### We can build a Gaussian Likelihood:
### $P\left(\left[\mathrm{Shots}\right] \mid\left[Goals\right]\right)=\frac{1}{\sqrt{2 \pi \Sigma_n^2}} \exp \left(\frac{-\left([\mathrm{Shots}]-\left[\overline{\mathrm{Shots}}\right]\right)^2}{2 \Sigma_n^2}\right)$
### with $\left[\overline{\mathrm{Shots}}\right] = m \cdot \mathrm{Goals} + q$  and $\Sigma^2=\vec{\nu}^T\left(S_n+\Lambda\right) \vec{\nu}$
### where $\vec{\nu}$ is the vector orthogonal to the mean relation ($\vec{\nu}$ = (-m,1) in case of a straight line)
### $S_n$ is the uncertainty tensor and $\Lambda$ the scatter tensor

### In the case of no error for goals and shots, the variance reduces to the scatter $\Sigma^2 = \lambda^2$
### We can sample the posterior distribution of the parameters $m, q, \lambda$ and characterise the scaling relation, including it into the model.


```python
import seaborn as sns

position_probabilities = {team_id: counts / np.sum(counts) for team_id, counts in position_counts.items()}
team_list = list(position_probabilities.keys())
team_name_list = [teams.loc[teams['TeamID'] == team_id, 'TeamName'].values[0] for team_id in position_probabilities.keys()]

prob_matrix = np.array([position_probabilities[team_id] for team_id in team_list])

positions = range(1, 29)

plt.figure(figsize=(30, 20))
sns.heatmap(prob_matrix, cmap="YlGnBu", annot=True, annot_kws={"size": 15},
            fmt=".2f", xticklabels=positions, yticklabels=team_name_list)

plt.xlabel("Position", fontsize=18)
plt.ylabel("Team ID", fontsize=18)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

```

![png](/Users/federico/Library/CloudStorage/OneDrive-UniversityCollegeLondon/PhD/CV/federico-speranza.github.io/assets/images/post_football/output_55_0.png)
    


### The matrix above shows the probability of each team to end in each position of the final standing. 
### The position probabilities are calculated by counting how many times each team finishes in each position across simulations and dividing by the total number of simulations. While this empirical approach provides straightforward estimates, alternative methods could give more refined insights. For instance, using a regression-based probability estimation model which predict the probability of each position based on various features, could give a more precise estimate of the standings probabilities.

---
----
---

### Extra

### I explored the relation between goal scored and shot attemps to try to include this information in the model.
### There is correlation between the two variables, however the data show a considerable scatter.
#### (See the cells above for comments and possible approaches)


```python
plt.figure(figsize=(15,30))

for k in range(1,29):
    plt.subplot(7,4,k)
    score = []
    shots = []
    defence_a = []
    for row in results.loc[(results['SeasonID']==1)].itertuples():
        if row.HomeTeamID == k:
            a_id = row.AwayTeamID
            defence_a.append(teams_params.loc[teams_params['TeamID']==a_id, 'beta'].iloc[0])
            shots.append(row.HomeShots)
            score.append(row.HomeScore)
    

    score = pd.Series(score)
    shots = pd.Series(shots)
    corr = score.corr(shots)
    
    plt.title(f'Correlation={round(corr,2)}')
    plt.plot(shots, score, 'o', label=f'TeamID_Home={k}', color=f'C{k}')
    
    plt.xlabel('Shots')
    plt.ylabel('Score')
#     plt.legend()
    
plt.tight_layout()

```

![png](/Users/federico/Library/CloudStorage/OneDrive-UniversityCollegeLondon/PhD/CV/federico-speranza.github.io/assets/images/post_football/output_60_0.png)
    


-----
-----
