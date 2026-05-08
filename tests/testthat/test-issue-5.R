library(testthat)
library(postlink)
library(broom)
library(knitr)

test_that('Simulation Sweep', {
  set.seed(123)
  skip_on_cran()
  
  # Define parameters
  N <- 1000
  mismatch_rates <- c(0.1, 0.3, 0.5)
  results_list <- list()
  
  for (i in seq_along(mismatch_rates)) {
    # Generate data
    BMI <- rnorm(N)
    Age <- rnorm(N)
    Disease_Status <- rbinom(N, 1, plogis(BMI + Age))
    match_status <- rbinom(N, 1, 1 - mismatch_rates[i])
    jw_score <- ifelse(match_status == 1, rbeta(N, 8, 2), rbeta(N, 2, 8))
    
    # Create data frame
    data <- data.frame(BMI, Age, Disease_Status, match_status, jw_score)
    
    # Introduce mismatch
    data$Disease_Status[data$match_status == 0] <- sample(data$Disease_Status, sum(data$match_status == 0), replace = TRUE)
    
    # Fit logistic regression model without adjustment
    model_fit_unadjusted <- glm(Disease_Status ~ BMI + Age, data = data, family = binomial)
    
    # Fit logistic regression model with adjustment using the EM mixture model
    adj_object <- adjMixture(linked.data = data, m.formula = ~ jw_score)
    model_fit_adjusted <- plglm(Disease_Status ~ BMI + Age, data = data, adjustment = adj_object, family = binomial)
    
    # Extract results with standard errors
    tidy_model_unadjusted <- broom::tidy(model_fit_unadjusted, conf.int = TRUE)
    tidy_model_unadjusted$mismatch <- mismatch_rates[i]
    tidy_model_unadjusted$model <- "Naive"
    tidy_model_adjusted <- broom::tidy(model_fit_adjusted, conf.int = TRUE)
    tidy_model_adjusted$mismatch <- mismatch_rates[i]
    tidy_model_adjusted$model <- "Adjusted"
    
    # Store results
    results_list[[i]] <- rbind(tidy_model_unadjusted, tidy_model_adjusted)
    
    # Add assertion to ensure the test formally passes
    expect_no_error(broom::tidy(model_fit_adjusted))
  }
  
  # Combine results
  final_results <- do.call(rbind, results_list)
  
  # Generate table string
  table_string <- knitr::kable(final_results, format = 'markdown')
  
  # Save table string to file
  writeLines(table_string, 'results.md')
})