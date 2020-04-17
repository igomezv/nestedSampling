# nestedSampling

Toy nested sampling based on Skilling (2006) Nested Sampling for General Bayesian Computation. 

Prior mass:

<img src="https://render.githubusercontent.com/render/math?math=\int_{L(\theta) > \lambda} \pi (\theta) d \theta">

Evidence: 
<img src="https://render.githubusercontent.com/render/math?math=Z=\int_0^1 L(X) dX">

<img src="https://render.githubusercontent.com/render/math?math=<0 (X_{m+1}) < X_m < ... < X_2 < X_1 < 1 (X_0)">

-> Evidence: 

<img src="https://render.githubusercontent.com/render/math?math=Z = \sum_{i=1}^m w_i(t) L_i">

where:

<img src="https://render.githubusercontent.com/render/math?math=\omega \approx \Delta X">

Initial prior mass:

<img src="https://render.githubusercontent.com/render/math?math=X_0 = 1">


Remaining prior mass : 

<img src="https://render.githubusercontent.com/render/math?math=X_i = t_i X_{i-1}">

where : 

<img src="https://render.githubusercontent.com/render/math?math=Pr(t_i) = N t_i^{N-1} \in (0,1)">


where N is the number of live points and t_i the largest of N random numbers from Uniform(0, 1). 


<img src="https://render.githubusercontent.com/render/math?math=t_i \sim \beta(N,1)">

Mean:

<img src="https://render.githubusercontent.com/render/math?math=E(log t) = \frac{-1}{N}">

Std dev: 

<img src="https://render.githubusercontent.com/render/math?math=dev(log) = \frac{1}{N}">

Algorithm 1 Skilling's paper:

	1) Start with N points from prior;
	2) initialise Z = 0, X_0 = 1.
	3) Repeat for i = 1, 2, . . . , j;
	    3.1) record the lowest of the current likelihood values as 
	    L i ,

	    3.2) set X i = exp(−i/N ) (crude) or sample it to get 
	    uncertainty,

	    3.3) set w i = X i−1 − X i (simple) or 
	    (X i−1 − X i+1 )/2 (trapezoidal),

	    3.4) increment Z by L i w i ,
	    3.5) then replace point of lowest likelihood by 
	    new one drawn
	        from within L(θ) > L i , in proportion to t
	        he prior π(θ).
	4) Increment Z by N −1 (L(θ 1 ) + . . . + 
						L(θ N )) X j


