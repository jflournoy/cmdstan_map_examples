#'---
#'title: "parallelization of Stan programs using cmdstan and map_rect"
#'author: "John Flournoy"
#'date: "2/11/2020"
#'output: html_document
#'---
#'
#' # Some setup
#' 

library(rstan)
library(tidyverse)
library(tidybayes)

cmdstan_dir <- '../cmdstan/'
hlm_test_dir <- 'hlm_test'
cmdstan_path <- file.path(cmdstan_dir, hlm_test_dir)
set.seed(1219344)

# data {
#   int<lower=1> N;             // Number of observations
#   int<lower=1> J;             // Number of participants
#   int<lower=1,upper=J> id[N]; // Participant IDs
#   vector[N] X1;                // Manipulated variable
#   vector[N] X2;                // Mediator
#   // Priors
#   real prior_bs;
#   real prior_tau_bs;
#   real prior_lkj_shape;
#   real prior_sigma_y;
#   vector[N] Y;                // Continuous outcome
#   int<lower=0, upper=1> SIMULATE;
# }

#'
#' # Generate some fake data
#'
#' We can use the Stan program to generate our Y values, but we need to supply
#' it with the rest of the data structure.
#' 

n_per_j <- 30
J <- 1000

gen_priors <- list(
  bs = 1, tau_bs = 1,
  lkj_shape = 1,
  sigma_y = 1
)

out_data <- list()
out_data$J <- J
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

stan_rdump(ls(out_data), file.path(cmdstan_path, "test_hlm_input.R"), envir = list2env(out_data))

#'
#' To generate the data, make the hlm stan file by copying hlm.stan to the
#' cmdstand/hlm_test directory and then running `make hlm_test/hlm`. Then `cd
#' hlm_test` and run:
#' 
#' ``` 
#' time ./hlm sample num_samples=1 num_warmup=500 output file=sim.csv data file=test_hlm_input.R
#' ```
#' 

if(!file.exists('sim.rda')){
  sim_data <- rstan::read_stan_csv(file.path(cmdstan_path, 'sim.csv'))
  y_sim <- tidybayes::gather_draws(sim_data, `Y_sim.*`, regex = TRUE) %>%
    tidyr::extract(.variable, into = c('variable', 'id'), regex = '(Y_sim)\\.(.*)')
  params <- tidybayes::gather_draws(sim_data, `((tau_)*b[012]|.*_y|betas.*)`, regex = TRUE)
  save(sim_data, y_sim, params, file = 'sim.rda')
} else {
  load('sim.rda')
}

#'
#' # Estimate the model
#'
#' To estimate, we should use less informative priors.
#' 
 
default_priors <- list(
  bs = 1000, tau_bs = 50,
  lkj_shape = 1,
  sigma_y = 1
)

#'
#' ## Time a serialized single chain
#'
#' We'll use the same data, but add the simulated y values, and change our priors.
#' 

out_data_from_sim <- out_data
out_data_from_sim$Y <- y_sim$.value
out_data_from_sim$SIMULATE <- 0
out_data_from_sim$prior_bs <- default_priors$bs
out_data_from_sim$prior_tau_bs <- default_priors$tau_bs
out_data_from_sim$prior_lkj_shape <- default_priors$lkj_shape
out_data_from_sim$prior_sigma_y <- default_priors$sigma_y

stan_rdump(ls(out_data_from_sim), file.path(cmdstan_path, "test_hlm_input_to_est.R"), envir = list2env(out_data_from_sim))

#' 
#' To fit the model, go to cmdstan/hlm_test and run:
#' 
#' ```
#' time ./hlm sample num_samples=1000 num_warmup=1000 output file=fit.csv data file=test_hlm_input_to_est.R
#' ```
#'
#' ```
#' Threading is enabled. map_rect will run with at most 1 thread(s).
#' 
#' STAN_OPENCL is enabled. OpenCL supported functions will use:
#'   Platform: NVIDIA CUDA
#' Device: GeForce GTX 750 Ti
#' 
#' Gradient evaluation took 0.005109 seconds
#' 1000 transitions using 10 leapfrog steps per transition would take 51.09 seconds.
#' Adjust your expectations accordingly!
#'   
#' ...
#' 
#' Elapsed Time: 790.409 seconds (Warm-up)
#' 493.533 seconds (Sampling)
#' 1283.94 seconds (Total)
#' 
#' 
#' real	21m30.899s
#' user	21m30.708s
#' sys	0m0.544s
#' ```


if(!file.exists('sim_est.rds')){
  sim_est <- rstan::read_stan_csv(file.path(cmdstan_path, 'fit.csv'))
  saveRDS(sim_est, 'sim_est.rds')
} else {
  sim_est <- readRDS('sim_est.rds')
}
if(!file.exists('params_est.rds')){
  params_est <- tidybayes::gather_draws(sim_est, `((tau_)*b[012]|.*_y|betas.*)`, regex = TRUE)
  saveRDS(params_est, 'params_est.rds')
} else {
  params_est <- readRDS('params_est.rds')
}

params_est %>% group_by(.variable) %>%
  summarize(mean(.value)) %>%
  left_join(params)

#'
#' ## Time a map_rect'd single chain
#'  
#' Go to cmdstan/hlm_test and run:
#' 
#' ```
#' export STAN_NUM_THREADS=7
#' time ./hlm_map sample num_samples=1000 num_warmup=1000 output file=fit_map.csv data file=test_hlm_input_to_est.R
#' ```
#'  
#' ```
#' Threading is enabled. map_rect will run with at most 4 thread(s).
#' 
#' STAN_OPENCL is enabled. OpenCL supported functions will use:
#'   Platform: NVIDIA CUDA
#' Device: GeForce GTX 750 Ti
#' 
#' Gradient evaluation took 0.008023 seconds
#' 1000 transitions using 10 leapfrog steps per transition would take 80.23 seconds.
#' Adjust your expectations accordingly!
#'
#' ...
#'  
#' Elapsed Time: 1567.77 seconds (Warm-up)
#' 1027.71 seconds (Sampling)
#' 2595.48 seconds (Total)
#' 
#' 
#' real	14m15.530s
#' user	42m42.652s
#' sys	0m42.369s
#' ```

if(!file.exists('sim_est_map.rds')){
  sim_est_map <- rstan::read_stan_csv(file.path(cmdstan_path, 'fit_map.csv'))
  saveRDS(sim_est_map, 'sim_est_map.rds')
} else {
  sim_est_map <- readRDS('sim_est_map.rds')
}
if(!file.exists('params_est_map.rds')){
  params_est_map <- tidybayes::gather_draws(sim_est_map, `((tau_)*b[012]|.*_y|betas.*)`, regex = TRUE)
  saveRDS(params_est_map, 'params_est_map.rds')
} else {
  params_est_map <- readRDS('params_est_map.rds')
}

params_est_map %>% group_by(.variable) %>%
  summarize(mean(.value)) %>%
  left_join(params)

