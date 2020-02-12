functions{
  int[] count_per_id(int[] someIDs){
    int maxlev = max(someIDs);
    int counts[maxlev] = rep_array(0, maxlev);
    
    for(i in 1:size(someIDs)){
      counts[someIDs[i]] += 1;
    }
    return(counts);
  }
  int[] shard_ids(int[] someIDs, int nshards){
    int newids[size(someIDs)];
    for(i in 1:size(someIDs)){
      newids[i] = (someIDs[i] - 1) % nshards + 1;
    }
    return(newids);
  }
  int[] ids_per_shard(int maxID, int nshards){
    int ids_count[nshards] = rep_array(0,nshards);
    int shard_id;
    for(j in 1:maxID){
      shard_id = (j - 1) % nshards + 1;
      ids_count[shard_id] += 1; 
    }
    return(ids_count);
  }
  vector gi_glm(vector global, vector z_Us,
               real[] xr, int[] xi) {
    int K = 3;
    int J = xi[2];
    vector[3] betas = global[1:3];
    real sigma_y = global[4];
    vector[K] Tau = global[5:7];
    vector[K*(K-1)/2 + K] L_Omega_vec = global[8:(8+K*(K-1)/2 + K - 1)];
    matrix[K, J] z_U;
    matrix[K, K] L_Omega = rep_matrix(0, K, K);
    int M = size(xr) / K;
    int Mx = xi[1];
    vector[Mx] Y  = to_vector(xr[(M*0 + 1):(M*0 + Mx)]);
    vector[Mx] X1 = to_vector(xr[(M*1 + 1):(M*1 + Mx)]);
    vector[Mx] X2 = to_vector(xr[(M*2 + 1):(M*2 + Mx)]);
    int id[Mx] = xi[(M*0 + 3):(M*0 + Mx + 2)];
    
    matrix[J, K] U;
    vector[Mx] mu_y;
    real lp;
    real ll;
    
    for(j in 1:J){
      int start = (j - 1)*K + 1;
      int end = (j - 1)*K + K;
      z_U[,j] = z_Us[start:end];
    }
    for (k in 1:K){
      int km1 = k - 1;
      int start = k + (km1*(km1 - 1)/2);
      L_Omega[k, 1:k] = to_row_vector(L_Omega_vec[start:(start + km1)]);
    }
    U = (diag_pre_multiply(Tau, L_Omega) * z_U)';
    mu_y = (betas[2] + U[id, 2]) .* X1 +
           (betas[3] + U[id, 3]) .* X2 +
           (betas[1] + U[id, 1]);
    //to_vector(z_U) ~ normal(0, 1);
    lp = normal_lpdf(to_vector(z_U) | 0, 1);
    ll = normal_lpdf(Y | mu_y, sigma_y);
    return [lp + ll]';
  }
}
data {
    int<lower=1> N;             // Number of observations
    int<lower=1> J;             // Number of participants
    int<lower=1,upper=J> id[N]; // Participant IDs
    vector[N] X1;               // First independent variable
    vector[N] X2;               // Second IV
    // Priors
    real prior_bs;
    real prior_tau_bs;
    real prior_lkj_shape;
    real prior_sigma_y;
    vector[N] Y;                // Continuous outcome
    int<lower=0, upper=1> SIMULATE;
}
transformed data{
    int K = 3;                      // Number of predictors
    int nr = 3;                     // Number of real-valued variables
    int ni = 1;                     // Number of int-valued variables
    int<lower=1> nshards = 20;      // Number of shards
    int<lower=1, upper=nshards> shard[N] = shard_ids(id, nshards);
    int<lower=1, upper=N> counts[nshards] = count_per_id(shard);
    int<lower=1, upper=J> jcounts[nshards] = ids_per_shard(J, nshards);
    int<lower=1> M = max(counts);
    int<lower=1> s_r[nshards] = rep_array(1, nshards);
    int<lower=1> s_i[nshards] = rep_array(3, nshards);
    int xi[nshards, M*ni + 2];  
    real xr[nshards, M*nr]; 

    //create shards
    xi[,1] = counts;
    xi[,2] = jcounts;
    for (i in 1:N){
      int shard_i = shard[i];
      xr[shard_i, s_r[shard_i] + M*0] = Y[i];
      xr[shard_i, s_r[shard_i] + M*1] = X1[i];
      xr[shard_i, s_r[shard_i] + M*2] = X2[i];
      xi[shard_i, s_i[shard_i] + M*0] = (id[i] - 1) / nshards + 1;
      s_r[shard_i] += 1;
      s_i[shard_i] += 1;
    }
}
parameters{
    vector[K] betas;
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
    vector[K*(K-1)/2 + K] L_Omega_vec;
    vector[max(jcounts) * K] z_Us[nshards];
    {
      for (k in 1:K){
        int km1 = k - 1;
        int start = k + (km1*(km1 - 1)/2);
        L_Omega_vec[start:(start + km1)] = to_vector(L_Omega[k, 1:k]);
      }
      for(j in 1:J){
        int sh = (j - 1) % nshards + 1;
        int start = ((j - 1) / nshards) * K + 1;
        int end = ((j - 1) / nshards) * K + K ;
        z_Us[sh, start:end] = z_U[, j];
      } 
    }
}
model {
    // Means of linear models
    // Regression parameter priors
    betas ~ normal(0, prior_bs);
    sigma_y ~ exponential(prior_sigma_y);
    // SDs and correlation matrix
    Tau ~ cauchy(0, prior_tau_bs);   // u_b0
    L_Omega ~ lkj_corr_cholesky(prior_lkj_shape);
    // Allow vectorized sampling of varying effects via stdzd z_U
    
    // Data model
    if(SIMULATE == 0){
        target += sum(map_rect(gi_glm, append_row(append_row(betas, sigma_y), 
                                                  append_row(Tau, L_Omega_vec)), 
                               z_Us, xr, xi));
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