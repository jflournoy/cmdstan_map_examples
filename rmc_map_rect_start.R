library(rstan)
d <- read.csv( "../cmdstan_map_rect_tutorial/RedcardData.csv" , stringsAsFactors=FALSE )
table( d$redCards )
d2 <- d[ !is.na(d$rater1) , ]
out_data <- list( n_redcards=d2$redCards , n_games=d2$games , rating=d2$rater1 )
out_data$N <- nrow(d2)

stan_rdump(ls(out_data), "redcard_input.R", envir = list2env(out_data))
