 
# The SIR Model

For this problem, we consider a population of \( N \) individuals, divided into three subgroups:

- \( S(t) \): Susceptible individuals
- \( I(t) \): Infected individuals
- \( R(t) \): Removed individuals at time \( t \)

The following two equations model the epidemic:

\[ S + I \xrightarrow{k_1} 2I \]
\[ I \xrightarrow{k_2} R \]

The mean-field model for the evolution of the averages is given by the following equations:

\[ \frac{d \langle S \rangle}{dt} = -\frac{k_1}{\nu} \langle S \rangle \langle I \rangle \]
\[ \frac{d \langle I \rangle}{dt} = \frac{k_1}{\nu} \langle S \rangle \langle I \rangle - k_2 \langle I \rangle \]

## Hybrid Modelling

We aim to model this system using both a stochastic algorithm paired with the ODE. To implement this hybrid method, we define two different particles (continuous and discrete) for each species. This hybrid model was pioneered by Kynaston and Yates. This SIR model is a novel case for this particular hybrid technique.

We divide the chemicals into discrete-continuous parts, i.e., \((S,I) \rightarrow (D_S,D_I,C_S,C_I)\), where the prefix corresponds to either **D**iscrete or **C**ontinuous, and the suffix corresponds to being **I**nfected or **S**usceptible.

### Reaction One: \( S + I \rightarrow 2I \)

This is a two-species reaction, giving three possible reactions for the discrete chemical interactions: one involving two discrete particles and two reactions involving a combination of discrete and continuous particles. Note that we do not consider two continuous particles reacting, as this is handled by the ordinary differential equation. This can be split into the following three equations:

\[ (1.1) \quad D_S + D_I \rightarrow 2D_I \]

\[ (1.2) \quad D_S + C_I \rightarrow 2D_I \]

\[ (1.3) \quad C_S + D_I \rightarrow 2D_I \]

### Reaction Two: \( I \rightarrow R \)

According to the laws of the paper, a continuous particle cannot become discrete if the product is different. Therefore, this gives us only one case:

\[ (1.4) \quad D_I \rightarrow D_S \]

This then gives us a total of four reactions controlling the dynamics of the system. There are four more reactions involved to be considered; the conversion reactions. This will be explained in the following section.

### Conversion Reactions

We have the following conversion reactions:

\[ (1.5) \quad C_S \rightarrow D_S \]

\[ (1.6) \quad C_I \rightarrow D_I \]

\[ (1.7) \quad D_S \rightarrow C_S \]

\[ (1.8) \quad D_I \rightarrow C_I \]

We can define the stoichiometric matrix from reactions (1.1) to (1.8) as the following:

\[ 
\left[ \begin{array}{cccccccc}
-1 &  0 & -1 &  0 &  1 &  0 & -1 &  0 \\
 1 &  1 &  2 & -1 &  0 &  1 &  0 & -1 \\
 0 & -1 &  0 &  0 & -1 &  0 &  1 &  0 \\
 0 &  0 & -1 &  0 &  0 & -1 &  0 &  1 \\
\end{array} \right]
\]

This matrix succinctly captures the dynamics of the system, showing how each reaction changes the quantities of the discrete and continuous susceptible and infected individuals.