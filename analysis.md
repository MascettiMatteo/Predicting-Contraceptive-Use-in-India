Predicting Contraceptive Use in India
================
Matteo Mascetti
25-11-2026

- [1. Setup and data](#1-setup-and-data)
- [2. Exploratory analysis](#2-exploratory-analysis)
- [3. Feature engineering](#3-feature-engineering)
  - [Region](#region)
  - [Age and fertility](#age-and-fertility)
  - [Reference levels](#reference-levels)
  - [Association between covariates](#association-between-covariates)
- [4. Data splitting](#4-data-splitting)
- [5. Variable selection](#5-variable-selection)
- [6. Non-linear effect of age](#6-non-linear-effect-of-age)
- [7. Models](#7-models)
  - [Model 0: benchmark (intercept
    only)](#model-0-benchmark-intercept-only)
  - [Model 1: GAM](#model-1-gam)
  - [Model 2: logistic regression, selected
    covariates](#model-2-logistic-regression-selected-covariates)
  - [Model 3: parsimonious logistic regression
    (`state + child + age`)](#model-3-parsimonious-logistic-regression-state--child--age)
  - [Model 4: PCR](#model-4-pcr)
  - [Model 5: Ridge](#model-5-ridge)
  - [Model 6: Lasso](#model-6-lasso)
  - [Model 7: Elastic Net (mixture =
    0.5)](#model-7-elastic-net-mixture--05)
  - [Model 8: Elastic Net with `state` x `child`
    interactions](#model-8-elastic-net-with-state-x-child-interactions)
  - [Model 9: Elastic Net with all pairwise
    interactions](#model-9-elastic-net-with-all-pairwise-interactions)
- [8. Equivalent degrees of freedom](#8-equivalent-degrees-of-freedom)
- [9. Model comparison](#9-model-comparison)
  - [Validation set](#validation-set)
  - [Test set](#test-set)
- [10. Final model: Lasso](#10-final-model-lasso)
  - [Interpretation](#interpretation)
  - [Future work](#future-work)
- [Session info](#session-info)

Predictive modelling of contraceptive use among ~18,000 Indian women.
This document reproduces the full analysis of the report
(`report/report_mascetti.pdf`): exploratory analysis, feature
engineering, model comparison and final model interpretation.

> **Before knitting**: put `training.csv` in `data/` (or change
> `data_path` below). Tuning chunks are cached, so the first knit is
> slow (the all-interactions Elastic Net is the heaviest part).

# 1. Setup and data

``` r
library(tidyverse)
library(tidymodels)
library(mgcv)
library(splines)

data_path <- "/Users/matteo/Desktop/APPUNTI MATTEO/3. Università/MAGISTRALE/1 ANNO/DATA MINING/training.csv"
india <- read.csv(data_path)
glimpse(india)
```

    ## Rows: 18,314
    ## Columns: 8
    ## $ ID            <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1…
    ## $ state         <chr> "Jammu & Kashmir", "Jammu & Kashmir", "Jammu & Kashmir",…
    ## $ age           <int> 49, 26, 33, 43, 38, 25, 25, 28, 43, 39, 35, 27, 36, 42, …
    ## $ education     <chr> "None", "Intermediate", "High", "None", "None", "None", …
    ## $ child         <chr> "More than 1", "More than 1", "More than 1", "More than …
    ## $ religion      <chr> "Muslim", "Muslim", "Muslim", "Muslim", "Muslim", "Musli…
    ## $ area          <chr> "rural", "rural", "rural", "rural", "rural", "rural", "r…
    ## $ contraceptive <int> 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0,…

`contraceptive`, `child`, `education`, `religion` and `area` are stored
as numbers or strings but are categorical, so they are converted to
factors.

``` r
india <- india %>%
  mutate(across(c(contraceptive, child, education, religion, area), as.factor))

# helper: drop covariates by name, returns kept indexes
idx.drop <- function(data, cov_name) which(!colnames(data) %in% cov_name)
```

# 2. Exploratory analysis

``` r
table(india$state)
```

    ## 
    ##     Andhra Pradesh  Arunachal Pradesh              Assam              Bihar 
    ##                928                 73                447                707 
    ##         Chandigarh       Chhattisgarh Dadra+Nagar Haveli        Daman & Diu 
    ##                 33                563                 26                 27 
    ##                Goa            Gujarat            Haryana   Himachal Pradesh 
    ##                 63                799                848                691 
    ##    Jammu & Kashmir          Jharkhand          Karnataka             Kerala 
    ##                300                373               1596                674 
    ##     Madhya Pradesh        Maharashtra            Manipur          Meghalaya 
    ##               1332               1528                 24                 25 
    ##            Mizoram           Nagaland       NCT of Delhi             Orissa 
    ##                  8                 23                345                900 
    ##        Pondicherry             Punjab          Rajasthan             Sikkim 
    ##                 52                773               1221                 41 
    ##         Tamil Nadu            Tripura      Uttar Pradesh        Uttarakhand 
    ##                745                100               1712                203 
    ##        West Bengal 
    ##               1134

``` r
table(india$education)
```

    ## 
    ##         High Intermediate          Low         None 
    ##         2376         6274         2982         6682

``` r
table(india$child)
```

    ## 
    ##           0           1 More than 1 
    ##        1118        2752       14444

``` r
table(india$religion)
```

    ## 
    ## Christian     Hindu    Muslim     Other 
    ##       401     14959      2233       721

``` r
table(india$area)
```

    ## 
    ## rural urban 
    ## 12133  6181

``` r
# missing values
india %>% summarise(across(everything(), ~ sum(is.na(.x))))
```

    ##   ID state age education child religion area contraceptive
    ## 1  0     0   0         0     0        0    0             0

``` r
# response
prop.table(table(india$contraceptive))
```

    ## 
    ##         0         1 
    ## 0.2661352 0.7338648

Contraceptive use against each categorical covariate (row proportions):

``` r
cat.var <- india %>% select(-ID, -age, -contraceptive)

plot_response <- function(var_name) {
  tab <- table(cat.var[[var_name]], india$contraceptive)
  df_plot <- as.data.frame(round(prop.table(tab, 1), 4)) %>%
    rename(category = Var1, contraceptive = Var2, proportion = Freq) %>%
    mutate(contraceptive = factor(contraceptive,
                                  labels = c("No contraceptive", "Contraceptive")))

  ggplot(df_plot, aes(x = category, y = proportion, fill = contraceptive)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = scales::percent(proportion, accuracy = 0.1)),
              position = position_stack(vjust = 0.5),
              size = 3, color = "white", fontface = "bold", check_overlap = TRUE) +
    scale_fill_manual(values = c("No contraceptive" = "skyblue",
                                 "Contraceptive" = "orange")) +
    scale_y_continuous(labels = scales::percent_format()) +
    labs(title = paste("Contraceptive Distribution by", var_name),
         x = var_name, y = "Proportion", fill = NULL) +
    theme_minimal(base_size = 13) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
          legend.position = "top", panel.grid.major.x = element_blank()) +
    coord_flip()
}

plot_response("state")
```

![](figures/eda-state-1.png)<!-- -->

``` r
for (v in setdiff(colnames(cat.var), "state")) print(plot_response(v))
```

![](figures/eda-other-vars-1.png)<!-- -->![](figures/eda-other-vars-2.png)<!-- -->![](figures/eda-other-vars-3.png)<!-- -->![](figures/eda-other-vars-4.png)<!-- -->

Some states show a very low use of contraceptives. They are all in the
north-east of the country (between China, Bangladesh and Myanmar):
Mizoram, Manipur, Meghalaya, Nagaland, plus Assam.

``` r
focus_states <- c("Assam", "Mizoram", "Manipur", "Meghalaya", "Nagaland")
india.focus <- india %>% filter(state %in% focus_states)

# observations per state and number of children
table(india.focus$state)
```

    ## 
    ##     Assam   Manipur Meghalaya   Mizoram  Nagaland 
    ##       447        24        25         8        23

``` r
india.focus %>% with(table(state, child)) %>% prop.table(1) %>% round(3)
```

    ##            child
    ## state           0     1 More than 1
    ##   Assam     0.096 0.244       0.660
    ##   Manipur   0.042 0.208       0.750
    ##   Meghalaya 0.040 0.120       0.840
    ##   Mizoram   0.125 0.375       0.500
    ##   Nagaland  0.000 0.130       0.870

``` r
as.data.frame(prop.table(table(india.focus$state, india.focus$religion), 1)) %>%
  rename(state = Var1, religion = Var2, proportion = Freq) %>%
  ggplot(aes(x = state, y = proportion, fill = religion)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = scales::percent(proportion, accuracy = 0.1)),
            position = position_stack(vjust = 0.5), size = 3,
            color = "white", fontface = "bold", check_overlap = TRUE) +
  scale_fill_manual(values = c("royalblue", "#F28C38", "chartreuse4", "deeppink4")) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "State Focus Distribution for: religion",
       x = "State", y = "Proportion", fill = "religion") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
        legend.position = "top", panel.grid.major.x = element_blank()) +
  coord_flip()
```

![](figures/eda-focus-religion-1.png)<!-- -->

The religious landscape of these states is varied, with no clear
similarity. Checking the neighbouring states:

``` r
india %>%
  filter(state %in% c("Tripura", "Arunachal Pradesh")) %>%
  with(table(state, contraceptive)) %>%
  prop.table(1) %>% round(3)
```

    ##                    contraceptive
    ## state                   0     1
    ##   Arunachal Pradesh 0.521 0.479
    ##   Tripura           0.260 0.740

The low use in the north-east is therefore probably due, at least in
part, to the **small sample size** of these states rather than to a real
territorial pattern. This motivates grouping states into macro-areas.

# 3. Feature engineering

## Region

Macro-areas: North, South, East, West, North-East (kept apart because of
its geographic relevance) and Union Territories (governed directly by
the central government in New Delhi).

``` r
make_region <- function(df) {
  df %>% mutate(region = case_when(
    state %in% c("Himachal Pradesh", "Punjab", "Uttarakhand", "Haryana",
                 "Uttar Pradesh") ~ "North",
    state %in% c("Bihar", "Orissa", "West Bengal", "Jharkhand",
                 "Chhattisgarh") ~ "East",
    state %in% c("Assam", "Mizoram", "Manipur", "Meghalaya", "Nagaland", "Tripura",
                 "Arunachal Pradesh", "Sikkim") ~ "North East",
    state %in% c("Rajasthan", "Gujarat", "Maharashtra", "Madhya Pradesh",
                 "Goa") ~ "West",
    state %in% c("Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu") ~ "South",
    state %in% c("Chandigarh", "Dadra+Nagar Haveli", "Daman & Diu",
                 "Jammu & Kashmir", "NCT of Delhi", "Pondicherry") ~ "Union Territory"
  ))
}

india <- india %>% make_region() %>% mutate(state = as.factor(state),
                                            region = as.factor(region))
any(is.na(india$region))
```

    ## [1] FALSE

## Age and fertility

``` r
age.median <- median(india$age)
age.mean <- mean(india$age)

ggplot(india, aes(age)) +
  geom_histogram(aes(y = after_stat(density)), bins = 35, fill = "white", color = "grey40") +
  geom_vline(xintercept = age.median, color = "orange", linetype = 3, linewidth = 1) +
  geom_vline(xintercept = age.mean, color = "purple", linetype = 3, linewidth = 1) +
  labs(title = "Age distribution (orange: median, purple: mean)", x = "Age") +
  theme_minimal()
```

![](figures/age-hist-1.png)<!-- -->

The median age is 35, well above the national figure (23.6) because the
sample only includes women from fertile age (15) onwards. Indian women
experience faster ovarian ageing than European women, so the age bands
are recalibrated:

| Band | Ages  | Meaning               |
|------|-------|-----------------------|
| SF   | 15-19 | Start of fertility    |
| PK   | 20-25 | Peak                  |
| SD   | 26-30 | Slight decrease       |
| CD   | 31-35 | Considerable decrease |
| SB   | 36-50 | Subfertility          |

``` r
make_fertility <- function(df) {
  df %>% mutate(fertility = case_when(
    age >= 15 & age <= 19 ~ "SF",
    age >= 20 & age <= 25 ~ "PK",
    age >= 26 & age <= 30 ~ "SD",
    age >= 31 & age <= 35 ~ "CD",
    age >= 36 & age <= 50 ~ "SB"
  ))
}

india <- india %>% make_fertility() %>% mutate(fertility = as.factor(fertility))
```

## Reference levels

``` r
india$contraceptive <- relevel(india$contraceptive, ref = "1")
india$region    <- relevel(india$region,    ref = "Union Territory")
india$area      <- relevel(india$area,      ref = "rural")
india$education <- relevel(india$education, ref = "High")
india$child     <- relevel(india$child,     ref = "0")
india$religion  <- relevel(india$religion,  ref = "Hindu")
india$fertility <- relevel(india$fertility, ref = "PK")

india <- india %>% select(-ID)   # ID has no predictive value
glimpse(india)
```

    ## Rows: 18,314
    ## Columns: 9
    ## $ state         <fct> Jammu & Kashmir, Jammu & Kashmir, Jammu & Kashmir, Jammu…
    ## $ age           <int> 49, 26, 33, 43, 38, 25, 25, 28, 43, 39, 35, 27, 36, 42, …
    ## $ education     <fct> None, Intermediate, High, None, None, None, None, None, …
    ## $ child         <fct> More than 1, More than 1, More than 1, More than 1, More…
    ## $ religion      <fct> Muslim, Muslim, Muslim, Muslim, Muslim, Muslim, Muslim, …
    ## $ area          <fct> rural, rural, rural, rural, rural, rural, rural, rural, …
    ## $ contraceptive <fct> 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0,…
    ## $ region        <fct> Union Territory, Union Territory, Union Territory, Union…
    ## $ fertility     <fct> SB, SD, CD, SB, SB, PK, PK, SD, SB, SB, CD, SD, SB, SB, …

## Association between covariates

Chi-squared tests between all pairs of categorical covariates:

``` r
x <- india %>% select(where(is.factor), -contraceptive)

combn(colnames(x), 2, simplify = FALSE) %>%
  map_dfr(function(p) {
    test <- chisq.test(table(x[[p[1]]], x[[p[2]]]))
    tibble(var1 = p[1], var2 = p[2],
           statistic = round(unname(test$statistic), 2),
           p.value = format.pval(test$p.value, eps = 0.001))
  })
```

    ## # A tibble: 21 × 4
    ##    var1      var2      statistic p.value
    ##    <chr>     <chr>         <dbl> <chr>  
    ##  1 state     education     2443. < 0.001
    ##  2 state     child          382. < 0.001
    ##  3 state     religion     11327. < 0.001
    ##  4 state     area          1598. < 0.001
    ##  5 state     region       91570  < 0.001
    ##  6 state     fertility      400. < 0.001
    ##  7 education child         1071. < 0.001
    ##  8 education religion       246. < 0.001
    ##  9 education area          1685. < 0.001
    ## 10 education region         380. < 0.001
    ## # ℹ 11 more rows

# 4. Data splitting

40% training, 30% validation, 30% test. Hyper-parameters are tuned on
the validation set; the test set is only used for the final comparison.

``` r
set.seed(123)
split <- initial_validation_split(india, prop = c(0.4, 0.3))
india.tr   <- training(split)
india.val  <- validation(split)
india.test <- testing(split)

c(train = nrow(india.tr), validation = nrow(india.val), test = nrow(india.test))
```

    ##      train validation       test 
    ##       7325       5494       5495

``` r
val_resample <- validation_set(split)
```

# 5. Variable selection

Stepwise selection (both directions) on AIC, comparing specifications
**with `state`** and **without `state`** (i.e. with `region` instead).

``` r
glm_null <- glm(contraceptive ~ 1, data = india.tr, family = binomial)
glm_full_step <- glm(contraceptive ~ ., data = india.tr, family = binomial)
glm_full_wo_state <- glm(contraceptive ~ . - state, data = india.tr, family = binomial)

bss <- stats::step(glm_null,
                   scope = list(lower = glm_null, upper = glm_full_step),
                   direction = "both", trace = 0)
bss.ws <- stats::step(glm_null,
                      scope = list(lower = glm_null, upper = glm_full_wo_state),
                      direction = "both", trace = 0)
```

``` r
models <- list(full = glm_full_step, best_subset = bss,
               full.ws = glm_full_wo_state, best_subset.WS = bss.ws)

tibble(model = names(models),
       AIC = sapply(models, AIC),
       n_cov = sapply(models, function(m) length(coef(m)) - 1))
```

    ## # A tibble: 4 × 3
    ##   model            AIC n_cov
    ##   <chr>          <dbl> <dbl>
    ## 1 full           6446.    51
    ## 2 best_subset    6446.    46
    ## 3 full.ws        6803.    19
    ## 4 best_subset.WS 6803.    19

``` r
formula(bss)
```

    ## contraceptive ~ child + state + fertility + religion + area + 
    ##     age + education

Models including `state` have a much lower AIC than models with
`region`. The two variables carry overlapping information
(collinearity), but the granularity of the state is more informative.
The base specification therefore keeps `state` and drops `region`.

# 6. Non-linear effect of age

A GAM is used to check whether `age` has a non-linear relationship with
the response.

``` r
m_gam_plot <- gam(contraceptive ~ s(age) + state + child + religion +
                    education + area + fertility,
                  data = india.tr, family = "binomial", method = "REML")
summary(m_gam_plot)
```

    ## 
    ## Family: binomial 
    ## Link function: logit 
    ## 
    ## Formula:
    ## contraceptive ~ s(age) + state + child + religion + education + 
    ##     area + fertility
    ## 
    ## Parametric coefficients:
    ##                               Estimate     Std. Error z value
    ## (Intercept)                    2.77442        0.35069   7.911
    ## stateArunachal Pradesh         1.09675        0.48915   2.242
    ## stateAssam                     1.52287        0.21577   7.058
    ## stateBihar                     0.59698        0.17740   3.365
    ## stateChandigarh               -1.08826        1.08987  -0.999
    ## stateChhattisgarh             -0.69641        0.21576  -3.228
    ## stateDadra+Nagar Haveli       -0.58592        0.89680  -0.653
    ## stateDaman & Diu              -1.02778        1.17941  -0.871
    ## stateGoa                      -0.69340        0.52293  -1.326
    ## stateGujarat                  -0.38690        0.18722  -2.067
    ## stateHaryana                  -0.24389        0.18070  -1.350
    ## stateHimachal Pradesh         -1.85298        0.25433  -7.286
    ## stateJammu & Kashmir          -0.01131        0.25200  -0.045
    ## stateJharkhand                 0.03455        0.22236   0.155
    ## stateKarnataka                -1.10892        0.16634  -6.667
    ## stateKerala                   -1.01100        0.22168  -4.561
    ## stateMadhya Pradesh           -1.41929        0.18506  -7.670
    ## stateMaharashtra              -1.26607        0.17937  -7.058
    ## stateManipur                   2.95845        1.09784   2.695
    ## stateMeghalaya                 2.96127        1.11325   2.660
    ## stateMizoram                  40.03954 33554432.00000   0.000
    ## stateNagaland                 41.45604 21221686.14265   0.000
    ## stateNCT of Delhi             -0.67509        0.27293  -2.474
    ## stateOrissa                   -1.20415        0.20195  -5.963
    ## statePondicherry              -0.24022        0.48931  -0.491
    ## statePunjab                   -1.25304        0.22866  -5.480
    ## stateRajasthan                -0.47604        0.16743  -2.843
    ## stateSikkim                  -39.99130 20234083.81773   0.000
    ## stateTamil Nadu               -0.42706        0.19472  -2.193
    ## stateTripura                  -1.13644        0.44645  -2.545
    ## stateUttar Pradesh            -0.08665        0.15114  -0.573
    ## stateUttarakhand              -1.55908        0.37189  -4.192
    ## stateWest Bengal              -1.54752        0.18937  -8.172
    ## child1                        -2.36052        0.17518 -13.475
    ## childMore than 1              -3.71238        0.17364 -21.380
    ## religionChristian              0.11926        0.25834   0.462
    ## religionMuslim                 0.42874        0.09731   4.406
    ## religionOther                  0.09119        0.19642   0.464
    ## educationIntermediate          0.08474        0.10783   0.786
    ## educationLow                  -0.07021        0.12969  -0.541
    ## educationNone                  0.19617        0.11589   1.693
    ## areaurban                     -0.25160        0.07586  -3.316
    ## fertilityCD                   -0.14502        0.30384  -0.477
    ## fertilitySB                   -0.08525        0.39329  -0.217
    ## fertilitySD                   -0.22201        0.19087  -1.163
    ## fertilitySF                    0.27808        0.33958   0.819
    ##                                     Pr(>|z|)    
    ## (Intercept)             0.000000000000002547 ***
    ## stateArunachal Pradesh              0.024952 *  
    ## stateAssam              0.000000000001689941 ***
    ## stateBihar                          0.000765 ***
    ## stateChandigarh                     0.318024    
    ## stateChhattisgarh                   0.001248 ** 
    ## stateDadra+Nagar Haveli             0.513537    
    ## stateDaman & Diu                    0.383517    
    ## stateGoa                            0.184843    
    ## stateGujarat                        0.038769 *  
    ## stateHaryana                        0.177121    
    ## stateHimachal Pradesh   0.000000000000319920 ***
    ## stateJammu & Kashmir                0.964205    
    ## stateJharkhand                      0.876536    
    ## stateKarnataka          0.000000000026197249 ***
    ## stateKerala             0.000005100406617228 ***
    ## stateMadhya Pradesh     0.000000000000017258 ***
    ## stateMaharashtra        0.000000000001683590 ***
    ## stateManipur                        0.007043 ** 
    ## stateMeghalaya                      0.007814 ** 
    ## stateMizoram                        0.999999    
    ## stateNagaland                       0.999998    
    ## stateNCT of Delhi                   0.013379 *  
    ## stateOrissa             0.000000002480313625 ***
    ## statePondicherry                    0.623477    
    ## statePunjab             0.000000042579264533 ***
    ## stateRajasthan                      0.004465 ** 
    ## stateSikkim                         0.999998    
    ## stateTamil Nadu                     0.028295 *  
    ## stateTripura                        0.010912 *  
    ## stateUttar Pradesh                  0.566424    
    ## stateUttarakhand        0.000027609355923745 ***
    ## stateWest Bengal        0.000000000000000303 ***
    ## child1                  < 0.0000000000000002 ***
    ## childMore than 1        < 0.0000000000000002 ***
    ## religionChristian                   0.644334    
    ## religionMuslim          0.000010539658691218 ***
    ## religionOther                       0.642453    
    ## educationIntermediate               0.431940    
    ## educationLow                        0.588228    
    ## educationNone                       0.090526 .  
    ## areaurban                           0.000912 ***
    ## fertilityCD                         0.633144    
    ## fertilitySB                         0.828399    
    ## fertilitySD                         0.244755    
    ## fertilitySF                         0.412854    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Approximate significance of smooth terms:
    ##          edf Ref.df Chi.sq    p-value    
    ## s(age) 3.646  4.648  31.05 0.00000998 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## R-sq.(adj) =  0.291   Deviance explained = 24.7%
    ## -REML = 3146.9  Scale est. = 1         n = 7325

``` r
plot(m_gam_plot, select = 1, scheme = 1)
```

![](figures/gam-explore-1.png)<!-- -->

`age` has about 3.6 estimated degrees of freedom: the relationship is
non-linear. In the following models `age` enters through a **natural
cubic spline with 4 df**.

# 7. Models

All models are compared on cross-entropy (log loss).

``` r
glm_model <- logistic_reg() %>% set_engine("glm") %>% set_mode("classification")
my_metrics <- metric_set(mn_log_loss)

# base recipe: natural spline on age
gam_recipe <- recipe(contraceptive ~ ., data = india.tr) %>% step_ns(age, deg_free = 4)

# validation log loss of a fitted workflow
val_ll <- function(fit) {
  augment(fit, new_data = india.val) %>%
    mn_log_loss(truth = contraceptive, .pred_1, event_level = "first") %>%
    pull(.estimate)
}

# tune on the validation set, refit best model on the training set
tune_and_fit <- function(wf, grid) {
  res  <- tune_grid(wf, resamples = val_resample, grid = grid, metrics = my_metrics)
  best <- select_best(res, metric = "mn_log_loss")
  list(tuning = res, best = best,
       fit = finalize_workflow(wf, best) %>% fit(data = india.tr))
}

# non-zero coefficients of a penalised model
nonzero <- function(fit) {
  fit %>% extract_fit_parsnip() %>% tidy() %>%
    filter(estimate != 0, term != "(Intercept)")
}

# glmnet workflow builder
glmnet_wf <- function(recipe, mixture) {
  workflow() %>%
    add_recipe(recipe) %>%
    add_model(logistic_reg(penalty = tune(), mixture = mixture) %>%
                set_engine("glmnet"))
}

dummy_recipe <- gam_recipe %>% step_dummy(all_factor_predictors())
```

## Model 0: benchmark (intercept only)

``` r
m0 <- workflow() %>%
  add_recipe(recipe(contraceptive ~ 1, data = india.tr)) %>%
  add_model(glm_model) %>% fit(india.tr)
m0.ll <- val_ll(m0)
m0.ll
```

    ## [1] 0.5847274

## Model 1: GAM

``` r
wf_gam <- workflow() %>%
  add_recipe(recipe(contraceptive ~ ., data = india.tr) %>% step_rm(region)) %>%
  add_model(gen_additive_mod(select_features = FALSE) %>%
              set_engine("mgcv") %>% set_mode("classification"),
            formula = contraceptive ~ s(age) + state + child + religion +
              education + area + fertility)

m_gam_simple <- fit(wf_gam, data = india.tr)
gam.ll <- val_ll(m_gam_simple)
gam.ll
```

    ## [1] 0.4709102

## Model 2: logistic regression, selected covariates

``` r
glm_full <- workflow() %>%
  add_recipe(gam_recipe %>% step_rm("region")) %>%
  add_model(glm_model) %>% fit(india.tr)

glmfull.ll <- val_ll(glm_full)
glmfull.ll
```

    ## [1] 0.4651119

## Model 3: parsimonious logistic regression (`state + child + age`)

The choice is motivated by the documented target-driven sterilization
campaigns in several states (e.g. Bihar, Madhya Pradesh), aimed mostly
at women who already had more than one child.

``` r
glm_simple <- workflow() %>%
  add_recipe(recipe(contraceptive ~ state + child + age, data = india.tr) %>%
               step_dummy(all_factor_predictors()) %>%
               step_ns(age, deg_free = 4)) %>%
  add_model(glm_model) %>% fit(india.tr)

glmsmp.ll <- val_ll(glm_simple)
glmsmp.ll
```

    ## [1] 0.4662089

## Model 4: PCR

``` r
wf_pcr <- workflow() %>%
  add_recipe(dummy_recipe %>% step_normalize(all_predictors()) %>%
               step_pca(all_predictors(), num_comp = tune())) %>%
  add_model(glm_model)

pcr <- tune_and_fit(wf_pcr, tibble(num_comp = 1:35))
autoplot(pcr$tuning, metric = "mn_log_loss") + theme_bw()
```

![](figures/m-pcr-1.png)<!-- -->

``` r
show_best(pcr$tuning, metric = "mn_log_loss", n = 3)
```

    ## # A tibble: 3 × 7
    ##   num_comp .metric     .estimator  mean     n std_err .config         
    ##      <int> <chr>       <chr>      <dbl> <int>   <dbl> <chr>           
    ## 1       33 mn_log_loss binary     0.499     1      NA pre33_mod0_post0
    ## 2       35 mn_log_loss binary     0.500     1      NA pre35_mod0_post0
    ## 3       25 mn_log_loss binary     0.502     1      NA pre25_mod0_post0

``` r
pcr.ll <- val_ll(pcr$fit)
```

## Model 5: Ridge

``` r
lambda_ridge <- exp(seq(-8, 5, length.out = 100))

ridge <- tune_and_fit(glmnet_wf(dummy_recipe, mixture = 0),
                      tibble(penalty = lambda_ridge))
autoplot(ridge$tuning, metric = "mn_log_loss") + theme_bw()
```

![](figures/m-ridge-1.png)<!-- -->

``` r
show_best(ridge$tuning, metric = "mn_log_loss", n = 3)
```

    ## # A tibble: 3 × 7
    ##    penalty .metric     .estimator  mean     n std_err .config          
    ##      <dbl> <chr>       <chr>      <dbl> <int>   <dbl> <chr>            
    ## 1 0.000335 mn_log_loss binary     0.463     1      NA pre0_mod001_post0
    ## 2 0.000383 mn_log_loss binary     0.463     1      NA pre0_mod002_post0
    ## 3 0.000436 mn_log_loss binary     0.463     1      NA pre0_mod003_post0

``` r
ridge.ll <- val_ll(ridge$fit)
```

## Model 6: Lasso

``` r
lambda_grid <- exp(seq(-8, 0, length.out = 100))

lasso <- tune_and_fit(glmnet_wf(dummy_recipe, mixture = 1),
                      tibble(penalty = lambda_grid))
autoplot(lasso$tuning, metric = "mn_log_loss") + theme_bw()
```

![](figures/m-lasso-1.png)<!-- -->

``` r
show_best(lasso$tuning, metric = "mn_log_loss", n = 3)
```

    ## # A tibble: 3 × 7
    ##   penalty .metric     .estimator  mean     n std_err .config          
    ##     <dbl> <chr>       <chr>      <dbl> <int>   <dbl> <chr>            
    ## 1 0.00156 mn_log_loss binary     0.459     1      NA pre0_mod020_post0
    ## 2 0.00169 mn_log_loss binary     0.459     1      NA pre0_mod021_post0
    ## 3 0.00144 mn_log_loss binary     0.459     1      NA pre0_mod019_post0

``` r
lasso.ll <- val_ll(lasso$fit)
```

## Model 7: Elastic Net (mixture = 0.5)

``` r
en <- tune_and_fit(glmnet_wf(dummy_recipe, mixture = 0.5),
                   tibble(penalty = lambda_grid))
autoplot(en$tuning, metric = "mn_log_loss") + theme_bw()
```

![](figures/m-en-1.png)<!-- -->

``` r
show_best(en$tuning, metric = "mn_log_loss", n = 3)
```

    ## # A tibble: 3 × 7
    ##   penalty .metric     .estimator  mean     n std_err .config          
    ##     <dbl> <chr>       <chr>      <dbl> <int>   <dbl> <chr>            
    ## 1 0.00233 mn_log_loss binary     0.459     1      NA pre0_mod025_post0
    ## 2 0.00253 mn_log_loss binary     0.459     1      NA pre0_mod026_post0
    ## 3 0.00215 mn_log_loss binary     0.459     1      NA pre0_mod024_post0

``` r
en.ll <- val_ll(en$fit)
```

## Model 8: Elastic Net with `state` x `child` interactions

``` r
lambda_int <- exp(seq(-10, 0, length.out = 100))

en.int <- tune_and_fit(
  glmnet_wf(dummy_recipe %>%
              step_interact(terms = ~ starts_with("state"):starts_with("child")),
            mixture = 0.5),
  tibble(penalty = lambda_int))
en.ll.int <- val_ll(en.int$fit)
```

## Model 9: Elastic Net with all pairwise interactions

``` r
en.int.all <- tune_and_fit(
  glmnet_wf(dummy_recipe %>%
              step_interact(terms = ~ all_predictors():all_predictors()),
            mixture = 0.5),
  tibble(penalty = lambda_int))
en.ll.int.all <- val_ll(en.int.all$fit)

tibble(model = c("Elastic Net Two Interactions", "Elastic Net All Interactions"),
       n_cov = c(nrow(nonzero(en.int$fit)), nrow(nonzero(en.int.all$fit))),
       LogLoss = round(c(en.ll.int, en.ll.int.all), 4))
```

    ## # A tibble: 2 × 3
    ##   model                        n_cov LogLoss
    ##   <chr>                        <int>   <dbl>
    ## 1 Elastic Net Two Interactions    79   0.453
    ## 2 Elastic Net All Interactions   225   0.452

# 8. Equivalent degrees of freedom

For Ridge they are computed from the singular values of the standardised
design matrix: $df(\lambda) = \sum_j d_j^2 / (d_j^2 + \lambda)$. For the
other models they are the number of estimated (non-zero) coefficients;
for the GAM, the sum of its effective degrees of freedom.

``` r
prepped <- prep(dummy_recipe, training = india.tr)
X  <- bake(prepped, new_data = NULL, all_predictors()) %>% as.matrix()
d  <- svd(scale(X))$d
ridge_df <- function(lambda, d) sum(d^2 / (d^2 + lambda))
best_pen <- ridge$best$penalty

edf <- c(
  Benchmark = 1,
  GAM = sum(m_gam_plot$edf2),
  Full = nrow(tidy(glm_full)),
  Simple = nrow(tidy(glm_simple)),
  PCR = nrow(tidy(pcr$fit)),
  Ridge = ridge_df(best_pen, d),
  Lasso = nrow(nonzero(lasso$fit)),
  `Elastic Net` = nrow(nonzero(en$fit)),
  `Elastic Net Two Interactions` = nrow(nonzero(en.int$fit)),
  `Elastic Net All Interactions` = nrow(nonzero(en.int.all$fit))
)
edf <- ceiling(edf)
```

# 9. Model comparison

## Validation set

``` r
tibble(model = names(edf),
       eDoF = edf,
       validation_log_loss = round(c(m0.ll, gam.ll, glmfull.ll, glmsmp.ll, pcr.ll,
                                     ridge.ll, lasso.ll, en.ll, en.ll.int,
                                     en.ll.int.all), 4)) %>%
  arrange(desc(validation_log_loss))
```

    ## # A tibble: 10 × 3
    ##    model                         eDoF validation_log_loss
    ##    <chr>                        <dbl>               <dbl>
    ##  1 Benchmark                        1               0.585
    ##  2 PCR                             34               0.499
    ##  3 GAM                             51               0.471
    ##  4 Simple                          39               0.466
    ##  5 Full                            50               0.465
    ##  6 Ridge                           49               0.463
    ##  7 Elastic Net                     46               0.459
    ##  8 Lasso                           45               0.459
    ##  9 Elastic Net Two Interactions    79               0.453
    ## 10 Elastic Net All Interactions   225               0.452

## Test set

``` r
fitted_models <- list(
  Benchmark = m0, GAM = m_gam_simple, Simple = glm_simple, Full = glm_full,
  PCR = pcr$fit, Ridge = ridge$fit, Lasso = lasso$fit, `Elastic Net` = en$fit,
  `Elastic Net Two Interactions` = en.int$fit,
  `Elastic Net All Interactions` = en.int.all$fit
)

results <- imap_dfr(fitted_models, function(model, name) {
  augment(model, new_data = india.test) %>%
    mn_log_loss(truth = contraceptive, .pred_1, event_level = "first") %>%
    transmute(model = name, cross_entropy = round(.estimate, 3))
}) %>%
  mutate(eDoF = edf[model]) %>%
  arrange(desc(cross_entropy))

results
```

    ## # A tibble: 10 × 3
    ##    model                        cross_entropy  eDoF
    ##    <chr>                                <dbl> <dbl>
    ##  1 Benchmark                            0.581     1
    ##  2 PCR                                  0.493    34
    ##  3 GAM                                  0.465    51
    ##  4 Simple                               0.461    39
    ##  5 Full                                 0.46     50
    ##  6 Ridge                                0.458    49
    ##  7 Lasso                                0.453    45
    ##  8 Elastic Net                          0.453    46
    ##  9 Elastic Net Two Interactions         0.45     79
    ## 10 Elastic Net All Interactions         0.445   225

``` r
ggplot(results, aes(x = eDoF, y = cross_entropy, label = model)) +
  geom_point(size = 2.5, color = "orange") +
  ggrepel::geom_text_repel(size = 3) +
  labs(title = "Test cross-entropy vs. model complexity",
       x = "Equivalent degrees of freedom", y = "Test cross-entropy") +
  theme_minimal()
```

![](figures/comparison-plot-1.png)<!-- -->

Apart from PCR, all models have good and very similar predictive
performance. The Elastic Net with all interactions achieves the lowest
cross-entropy, but with a very large number of equivalent degrees of
freedom: this inflates the variance of the estimator and makes
interpretation hard.

**Final model: Lasso.** Predictions are almost identical to the Elastic
Net with two interactions, with far fewer degrees of freedom, which
favours interpretability.

# 10. Final model: Lasso

``` r
lasso$best
```

    ## # A tibble: 1 × 2
    ##   penalty .config          
    ##     <dbl> <chr>            
    ## 1 0.00156 pre0_mod020_post0

The selected penalty is very small, so the solution is close to the
unpenalised fit.

Coefficients retained by the Lasso, as log-odds and odds ratios:

``` r
nonzero(lasso$fit) %>%
  mutate(odds_ratio = exp(estimate)) %>%
  select(term, estimate, odds_ratio) %>%
  arrange(desc(estimate)) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3))) %>%
  print(n = Inf)
```

    ## # A tibble: 45 × 3
    ##    term                   estimate odds_ratio
    ##    <chr>                     <dbl>      <dbl>
    ##  1 state_Nagaland            3.00      20.2  
    ##  2 region_North.East         1.58       4.84 
    ##  3 state_Meghalaya           1.53       4.61 
    ##  4 state_Manipur             1.44       4.21 
    ##  5 state_Mizoram             1.33       3.79 
    ##  6 state_Bihar               1.05       2.85 
    ##  7 state_Jharkhand           0.485      1.62 
    ##  8 state_Jammu...Kashmir     0.417      1.52 
    ##  9 religion_Muslim           0.408      1.50 
    ## 10 fertility_SF              0.396      1.49 
    ## 11 state_Assam               0.382      1.46 
    ## 12 state_Uttar.Pradesh       0.378      1.46 
    ## 13 region_South              0.318      1.38 
    ## 14 state_Haryana             0.21       1.23 
    ## 15 education_None            0.145      1.16 
    ## 16 state_Pondicherry         0.057      1.06 
    ## 17 state_Gujarat             0.054      1.06 
    ## 18 religion_Christian        0.041      1.04 
    ## 19 education_Intermediate    0.019      1.02 
    ## 20 state_Goa                -0.004      0.996
    ## 21 fertility_CD             -0.027      0.974
    ## 22 education_Low            -0.084      0.92 
    ## 23 state_NCT.of.Delhi       -0.107      0.899
    ## 24 state_Daman...Diu        -0.129      0.879
    ## 25 state_Chhattisgarh       -0.138      0.871
    ## 26 age_ns_4                 -0.171      0.843
    ## 27 state_Tamil.Nadu         -0.214      0.807
    ## 28 fertility_SD             -0.215      0.807
    ## 29 state_Chandigarh         -0.231      0.794
    ## 30 area_urban               -0.236      0.79 
    ## 31 state_Orissa             -0.632      0.532
    ## 32 state_Punjab             -0.641      0.527
    ## 33 state_Maharashtra        -0.7        0.496
    ## 34 state_Kerala             -0.77       0.463
    ## 35 age_ns_2                 -0.773      0.462
    ## 36 state_Madhya.Pradesh     -0.844      0.43 
    ## 37 state_Karnataka          -0.878      0.416
    ## 38 state_Uttarakhand        -0.904      0.405
    ## 39 age_ns_1                 -0.949      0.387
    ## 40 state_West.Bengal        -0.965      0.381
    ## 41 state_Himachal.Pradesh   -1.25       0.286
    ## 42 state_Tripura            -2.05       0.128
    ## 43 child_X1                 -2.13       0.119
    ## 44 child_More.than.1        -3.48       0.031
    ## 45 state_Sikkim             -3.85       0.021

``` r
nonzero(lasso$fit) %>%
  mutate(sign = ifelse(estimate > 0, "Higher odds", "Lower odds"),
         term = fct_reorder(term, estimate)) %>%
  ggplot(aes(x = estimate, y = term, fill = sign)) +
  geom_col() +
  scale_fill_manual(values = c("Higher odds" = "orange", "Lower odds" = "skyblue")) +
  labs(title = "Lasso coefficients (log-odds)", x = "Coefficient", y = NULL, fill = NULL) +
  theme_minimal() +
  theme(legend.position = "top")
```

![](figures/lasso-plot-1.png)<!-- -->

## Interpretation

- **Fertility phase**: compared with peak fertility, women at the start
  of the fertile period have higher odds of using contraceptives, while
  those in the decrease phases have lower odds. The subfertility
  coefficient is shrunk to zero.
- **Children**: compared with women without children, odds of use are
  lower for women with at least one child, suggesting use is driven by
  childless women who want to delay or avoid a first pregnancy.
- **State**: Bihar and Uttar Pradesh, where sterilization targets were
  documented, have higher odds than Andhra Pradesh (reference).
- **Geography and religion**: lower odds in the South and North-East,
  and among Muslim women.
- **Education**: no particularly marked effect.

## Future work

Build a variable that explicitly flags the presence of coercive
family-planning policies by state, to isolate their effect from the
other socio-cultural determinants.

# Session info

``` r
sessionInfo()
```

    ## R version 4.5.2 (2025-10-31)
    ## Platform: x86_64-apple-darwin20
    ## Running under: macOS Sequoia 15.8.2
    ## 
    ## Matrix products: default
    ## BLAS:   /Library/Frameworks/R.framework/Versions/4.5-x86_64/Resources/lib/libRblas.0.dylib 
    ## LAPACK: /Library/Frameworks/R.framework/Versions/4.5-x86_64/Resources/lib/libRlapack.dylib;  LAPACK version 3.12.1
    ## 
    ## locale:
    ## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
    ## 
    ## time zone: Europe/Rome
    ## tzcode source: internal
    ## 
    ## attached base packages:
    ## [1] splines   stats     graphics  grDevices utils     datasets  methods  
    ## [8] base     
    ## 
    ## other attached packages:
    ##  [1] glmnet_5.0         Matrix_1.7-4       mgcv_1.9-4         nlme_3.1-168      
    ##  [5] yardstick_1.4.0    workflowsets_1.1.1 workflows_1.3.0    tune_2.1.0        
    ##  [9] tailor_0.1.0       rsample_1.3.2      recipes_1.3.2      parsnip_1.6.0     
    ## [13] modeldata_1.5.1    infer_1.1.0        dials_1.4.3        scales_1.4.0      
    ## [17] broom_1.0.12       tidymodels_1.5.0   lubridate_1.9.5    forcats_1.0.1     
    ## [21] stringr_1.6.0      dplyr_1.2.1        purrr_1.2.2        readr_2.2.0       
    ## [25] tidyr_1.3.2        tibble_3.3.1       ggplot2_4.0.3      tidyverse_2.0.0   
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] tidyselect_1.2.1    timeDate_4052.112   farver_2.1.2       
    ##  [4] S7_0.2.1            fastmap_1.2.0       digest_0.6.39      
    ##  [7] rpart_4.1.24        timechange_0.4.0    lifecycle_1.0.5    
    ## [10] survival_3.8-3      magrittr_2.0.4      compiler_4.5.2     
    ## [13] rlang_1.2.0         tools_4.5.2         utf8_1.2.6         
    ## [16] yaml_2.3.12         data.table_1.18.2.1 knitr_1.51         
    ## [19] labeling_0.4.3      DiceDesign_1.10     RColorBrewer_1.1-3 
    ## [22] withr_3.0.2         nnet_7.3-20         grid_4.5.2         
    ## [25] sparsevctrs_0.3.6   future_1.70.0       iterators_1.0.14   
    ## [28] globals_0.19.1      MASS_7.3-65         cli_3.6.6          
    ## [31] rmarkdown_2.30      generics_0.1.4      otel_0.2.0         
    ## [34] rstudioapi_0.18.0   future.apply_1.20.2 tzdb_0.5.0         
    ## [37] parallel_4.5.2      vctrs_0.7.1         hardhat_1.4.3      
    ## [40] hms_1.1.4           ggrepel_0.9.8       listenv_0.10.1     
    ## [43] foreach_1.5.2       gower_1.0.2         glue_1.8.0         
    ## [46] parallelly_1.47.0   codetools_0.2-20    shape_1.4.6.1      
    ## [49] stringi_1.8.7       gtable_0.3.6        pillar_1.11.1      
    ## [52] furrr_0.4.0         htmltools_0.5.9     ipred_0.9-15       
    ## [55] lava_1.9.1          R6_2.6.1            evaluate_1.0.5     
    ## [58] lattice_0.22-7      backports_1.5.0     class_7.3-23       
    ## [61] Rcpp_1.1.1          prodlim_2026.03.11  xfun_0.56          
    ## [64] pkgconfig_2.0.3
