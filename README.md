# The SIR Model


For this particular problem we are considering a population of N individuals, divided into three subgroups.

- $S(t)$: Suceptible individuals
- $I(t)$: Infected 
- $R(t)$: Removed individuals at time $t$

The following two equations model the Epidemic:

$$
S+I \rightarrow^{k_1} 2I \\
I \rightarrow^{k_2} R
$$

The mean-field model for the evolution of the averages are given by the following equations:

$$
\frac{d <S>}{dt} = -\frac{k_1}{\nu}<S><I> \\
\frac{d <I>}{dt} = \frac{k_1}{\nu}<S><I>-k_2<I>
$$

## Hybrid modelling

We aim to model this system using both a stochastic algorithm paired with the ODE. In order to implement this Hybrid-method we must define two different particles (continious and discrete) for each species. Note this Hybrid Model was Pioneered by Kynaston and Yates. This SIR is a novel modelling case for this particular hybrid technique. 

We divide the chemicals up into discrete-continious parts. Ie $(S,I) \rightarrow (D_S,D_I,C_S,C_I)$ in which the prefix corresponds to either **D**iscrete or **C**ontinious, and the suffix corresponding to being **I**nfected or **S**uceptible. 

### Reaction one: $S+I \rightarrow 2I$

This is a two-species reaction, this gives three possible reactions for the discrete chemical interactions. One reaction involving two discrete particles, and two reactions in which a combination of discrete and Continious particles react. Note we do not consider two continious particles reacting as this is handled by the Ordinary differential equation. This can thus be split into the following three equations. 

$$
(1.1) \quad D_s + D_I \rightarrow 2D_I
$$

$$
(1.2) \quad D_s + C_I \rightarrow 2D_I
$$

$$
(1.3) \quad C_s + D_I \rightarrow 2D_I
$$

### Reaction two: $I \rightarrow R$

By the laws of the paper then a continious particle cannot become a discrete IF the product is different. Therefore this gives us only one case:

$$
(1.4) \quad D_I \rightarrow D_S
$$

This then gives us a total of four reactions which control the dynamics of the system. 

There are four more reactions involved to be considered; the conversion reactions. This will be explained in the following section

### Conversion reactions

We have the following conversion reactions: 

$$
(1.5) \quad  C_S \rightarrow D_S\\
$$
$$
(1.6) \quad C_I \rightarrow D_I\\
$$
$$
(1.7) \quad D_S \rightarrow C_S\\
$$

$$
(1.8) \quad D_I \rightarrow C_I\\
$$


We can define the stoichiometric matrix from Reactions $1-8$ as the following:
$$
\left[ \begin{array}{cccccccc}
 -1 &  0 & -1 &  0 &  1 &  0 & -1 &  0 \\
  1 &  1 &  2 & -1 &  0 &  1 &  0 & -1 \\
  0 & -1 &  0 &  0 & -1 &  0 &  1 &  0 \\
  0 &  0 & -1 &  0 &  0 & -1 &  0 &  1 \\
\end{array} \right]
$$



 