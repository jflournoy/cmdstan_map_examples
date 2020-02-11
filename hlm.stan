// Stan code for multilevel mediation model

data {
    int<lower=1> N;             // Number of observations
    int<lower=1> J;             // Number of participants
    int<lower=1,upper=J> id[N]; // Participant IDs
    vector[N] X1;                // Manipulated variable
    vector[N] X2;                // Mediator
    // Priors
    real prior_bs;
    real prior_tau_bs;
    real prior_lkj_shape;
    real prior_sigma_y;
    vector[N] Y;                // Continuous outcome
    int<lower=0, upper=1> SIMULATE;
}
transformed data{
    int K;                      // Number of predictors
    K = 3;
}
parameters{
    // Regression Y on X and M
    vector[K] betas;
    // Regression M on X

    // Correlation matrix and SDs of participant-level varying effects
    cholesky_factor_corr[K] L_Omega;
    vector<lower=0>[K] Tau;

    // Standardized varying effects
    matrix[K, J] z_U;
    real<lower=0> sigma_y;      // Residual
}
transformed parameters {
    // Participant-level varying effects
    matrix[J, K] U;
    U = (diag_pre_multiply(Tau, L_Omega) * z_U)';
}
model {
    // Means of linear models
    vector[N] mu_y;
    // Regression parameter priors
    betas ~ normal(0, prior_bs);
    sigma_y ~ exponential(prior_sigma_y);
    // SDs and correlation matrix
    Tau ~ cauchy(0, prior_tau_bs);   // u_b0
    L_Omega ~ lkj_corr_cholesky(prior_lkj_shape);
    // Allow vectorized sampling of varying effects via stdzd z_U
    to_vector(z_U) ~ normal(0, 1);

    // Regressions
    mu_y = (betas[2] + U[id, 2]) .* X1 +
           (betas[3] + U[id, 3]) .* X2 +
           (betas[1] + U[id, 1]);
    // Data model
    if(SIMULATE == 0){
        Y ~ normal(mu_y, sigma_y);
    }
}
generated quantities{
    matrix[K, K] Omega;         // Correlation matrix
    matrix[K, K] Sigma;         // Covariance matrix

    // Person-specific mediation parameters
    vector[J] u_b0;
    vector[J] u_b1;
    vector[J] u_b2;

    // Re-named tau parameters for easy output
    real tau_b0;
    real tau_b1;
    real tau_b2;
    
    real Y_sim[N];

    tau_b0 = Tau[1];
    tau_b1 = Tau[2];
    tau_b2 = Tau[3];

    Omega = L_Omega * L_Omega';
    Sigma = quad_form_diag(Omega, Tau);

    u_b0 = betas[1] + U[, 1];
    u_b1 = betas[2] + U[, 2];
    u_b2 = betas[3] + U[, 3];
    
    {
        vector[N] mu_y;
        if(SIMULATE == 1){
            mu_y = (betas[2] + U[id, 2]) .* X1 +
                   (betas[3] + U[id, 3]) .* X2 +
                   (betas[1] + U[id, 1]);
            Y_sim = normal_rng(mu_y, sigma_y);
        }
    }
}