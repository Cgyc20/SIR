

# Analysis directory

This file will describe what each python file does. 

## time_test.py

This runs the HybridModel (using Normal thresholds) and SISsimulation (Using Kirkwood)

This models the SIS system over a range of initial discrete suceptible value $(D_s = 0, 50, 100,....,200)$ so on, with either a variable or threshold or fixed threshold. 

This then will output the time for iteration for the ODE, SSA and HybridMethod and plot this. 




## accuracy_test.py


Input: Total simulations, The $D_I(0)$ values (as a list)

Returns: The accuracy of the ODE method vs the accuracy of the hybrid (with respect to the SSA)

This then will plot the mean absolute error for the ODE and Hybrid relative to the SSA.

Returns:

$$
\frac{1}{N} \sum_{j=0}^{N-1} |I_j - (IH)_j|_{j = 0,N-1}
$$

$$
\frac{1}{N} \sum_{j=0}^{N-1} |I_j - (IC)_j|_{j = 0,N-1}
$$

For each $D_S(0) = 1,2,3...$ 



## Efficiency_diff_threshold.py

This calculates the efficiency of two instances of the Hybrid mean for a fixed threshold of conversion, versus a threshold which is different for the forward and backwards reaction.

Below threshold and above threshold can be set (two straddle the original threshold.)

In this case we have 'Total sims' which is how many simulations of the algorithm occur, and 'total repeats' which is how many times we repeat the total simulations. This is because for each repeat we calculate the time for the algorithm to run. 

## Accuracy_threshold_test

Input: First threshold, Last threshold (for conversion)

Returns: Mean and Max error over time between Hybrid method and SSA


In this python file we are testing how accurate the Hybrid method is as we raise the threshold for conversion, we compare this against the SSA stochastic infected value over time. We take the average as before, and also the maximum error. 

$$
\frac{1}{N} \sum_{j=0}^{N-1} |I_j - (IH)_j|_{j = 0,N-1}
$$

We should in theory reduce error as the threshold of conversion is raised. 







