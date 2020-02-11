library(rstan)
library(tidybayes)
library(tidyverse)

# data {
#   int<lower=1> N;             // Number of observations
#   int<lower=1> J;             // Number of participants
#   int<lower=1,upper=J> id[N]; // Participant IDs
#   vector[N] X1;                // Manipulated variable
#   vector[N] X2;                // Mediator
#   // Priors
#   real prior_b0;
#   real prior_b1;
#   real prior_b2;
#   real prior_tau_b0;
#   real prior_tau_b1;
#   real prior_tau_b2;
#   real prior_lkj_shape;
#   vector[N] Y;                // Continuous outcome
#   int<lower=0, upper=1> SIMULATE;
# }

n_per_j <- 30
gen_priors <- list(
  bs = 1, tau_bs = 1,
  lkj_shape = 1,
  sigma_y = 1
)

out_data <- list()
out_data$J <- 1000
out_data$N <- n_per_j * out_data$J
out_data$id <- rep(1:out_data$J, n_per_j)
out_data$X1 <- rnorm(out_data$N)
out_data$X2 <- rnorm(out_data$N)
out_data$prior_bs <- gen_priors$bs
out_data$prior_tau_bs <- gen_priors$tau_bs
out_data$prior_lkj_shape <- gen_priors$lkj_shape
out_data$prior_sigma_y <- gen_priors$sigma_y
out_data$Y <- rep(0, out_data$N)
out_data$SIMULATE <- 1

stan_rdump(ls(out_data), "../cmdstan/hlm_test/test_hlm_input.R", envir = list2env(out_data))

#time ./hlm sample num_samples=1 num_warmup=500 output file=sim.csv data file=test_hlm_input.R

sim_data <- rstan::read_stan_csv('~/code_new/cmdstan/hlm_test/sim.csv')

y_sim <- tidybayes::gather_draws(sim_data, `Y_sim.*`, regex = TRUE) %>%
  tidyr::extract(.variable, into = c('variable', 'id'), regex = '(Y_sim)\\.(.*)')

params <- tidybayes::gather_draws(sim_data, `((tau_)*b[012]|.*_y|betas.*)`, regex = TRUE)

default_priors <- list(
  bs = 1000, tau_bs = 50,
  lkj_shape = 1,
  sigma_y = 1
)

out_data_from_sim <- out_data
out_data_from_sim$Y <- y_sim$.value
out_data_from_sim$SIMULATE <- 0
out_data_from_sim$prior_bs <- default_priors$bs
out_data_from_sim$prior_tau_bs <- default_priors$tau_bs
out_data_from_sim$prior_lkj_shape <- default_priors$lkj_shape
out_data_from_sim$prior_sigma_y <- default_priors$sigma_y

stan_rdump(ls(out_data_from_sim), "../cmdstan/hlm_test/test_hlm_input_to_est.R", envir = list2env(out_data_from_sim))
#time ./hlm sample num_samples=1000 num_warmup=1000 output file=fit.csv data file=test_hlm_input_to_est.R
sim_est <- rstan::read_stan_csv('~/code_new/cmdstan/hlm_test/fit.csv')
params_est <- tidybayes::gather_draws(sim_est, `((tau_)*b[012]|.*_y|betas.*)`, regex = TRUE)
params_est %>% group_by(.variable) %>%
  summarize(mean(.value)) %>%
  left_join(params)
rstan::summary(sim_est)

# export STAN_NUM_THREADS=7
# time ./hlm_map sample num_samples=1000 num_warmup=1000 output file=fit_map.csv data file=test_hlm_input_to_est.R
sim_est_map <- rstan::read_stan_csv('~/code_new/cmdstan/hlm_test/fit_map.csv')
params_est_map <- tidybayes::gather_draws(sim_est_map, `((tau_)*b[012]|.*_y|betas.*)`, regex = TRUE)
params_est_map %>% group_by(.variable) %>%
  summarize(mean(.value)) %>%
  left_join(params)


# ids_count=rep(0, 20); 
# for(j in 1:110){ 
#   sid = (j-1) %% 20 + 1; 
#   ids_count[sid] = ids_count[sid] + 1;
# }

# nshards=20
# K=3
# Us=matrix(nrow=nshards, ncol = 6 * K + 1)
# U=matrix(data=1:(110*K), nrow = 110, ncol = 3)
# for(j in 1:110){
#   sh = (j - 1) %% nshards + 1;
#   message('j: ', j)
#   message('sh: ', sh)
#   start = ((j - 1) %/% nshards) * K + 2;
#   message('start: ', start)
#   end = ((j - 1) %/% nshards) * K + K + 1;
#   message('end: ', end)
#   Us[sh, start:end] = U[j, ];
# } 
