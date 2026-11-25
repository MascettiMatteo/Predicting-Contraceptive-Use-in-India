# Predicting Contraceptive Use in India

Predictive modelling project (Data Mining course, MSc) on the socio-demographic drivers of contraceptive use among Indian women. The goal is twofold: build a model with good out-of-sample predictive performance and keep it interpretable enough to read the socio-cultural factors behind the prediction.

📄 Full report (Italian) & 💻 R Code: Available upon request.

---

## Research question

> Which socio-cultural characteristics push Indian women to use contraceptive methods?

India became the world's most populous country in 2023. During the 2010s several states promoted population-control policies, often through target-driven and coercive sterilization campaigns (see [Human Rights Watch, 2012](https://www.hrw.org/news/2012/07/12/india-target-driven-sterilization-harming-women)). This context motivates the analysis, in particular the role of **state of residence** and **number of children**.

## Data

About 18,000 Indian women, with the following variables:

| Variable | Description |
|---|---|
| `contraceptive` | Binary response: 1 = uses contraceptives |
| `state` | State of residence (33 states / union territories, circa 2010) |
| `education` | None / Low / Intermediate / High |
| `area` | Urban / Rural |
| `child` | Number of children (categorical) |
| `religion` | Religion practised |
| `age` | Age (15–50) |

> The dataset comes from a university course and is **not included** in this repository.

## Methodology

**1. Exploratory analysis**
- Proportion tables and bar plots of contraceptive use against each categorical covariate.
- Five north-eastern states (Mizoram, Manipur, Meghalaya, Nagaland, Assam) show very low use. Checking the neighbouring states (Tripura, Arunachal Pradesh) suggests this is more likely due to small sample sizes than to a real territorial pattern.

**2. Feature engineering**
- `region`: states grouped into North, South, East, West, North-East and Union Territories (governed directly by the central government).
- `fertility`: age bands calibrated on the faster ovarian ageing of Indian women reported in the literature: SF (15–19), PK (20–25), SD (26–30), CD (31–35), SB (36–50).
- Factor levels re-referenced to the most interpretable baselines (e.g. Hindu, rural, no children, peak fertility).

**3. Data split**
- 40% training / 30% validation / 30% test.

**4. Variable selection**
- AIC-based selection on logistic regression. Models with `state` have a much lower AIC than models with `region` (6,500 vs 6,860), so `region` is dropped from the final specification.

**5. Models compared** (all built with `tidymodels`, evaluated with cross-entropy / log loss)

| # | Model |
|---|---|
| 0 | Benchmark (intercept only) |
| 1 | GAM (`mgcv`), used to detect the non-linear effect of `age` (~3.6 edf) |
| 2 | Logistic regression, selected covariates, `age` as natural cubic spline (4 df) |
| 3 | Parsimonious logistic regression: `state + age + child` |
| 4 | PCR (PCA + logistic regression) |
| 5 | Ridge |
| 6 | Lasso |
| 7 | Elastic Net (α = 0.5) |
| 8 | Elastic Net + `state × child` interactions |
| 9 | Elastic Net + all pairwise interactions |

Effective degrees of freedom are computed for every model (via the SVD for Ridge) to compare complexity fairly.

## Results

Test-set performance:

| Model | Cross-entropy | Equivalent df |
|---|---:|---:|
| Benchmark | 0.581 | 1 |
| PCR | 0.493 | 34 |
| GAM | 0.465 | 51 |
| Simple GLM | 0.461 | 50 |
| Full GLM | 0.460 | 39 |
| Ridge | 0.454 | 49 |
| Elastic Net | 0.453 | 46 |
| **Lasso (final model)** | **0.453** | **45** |
| Elastic Net, two interactions | 0.450 | 77 |
| Elastic Net, all interactions | 0.445 | 218 |

Elastic Net with all interactions has the lowest cross-entropy, but it needs ~218 equivalent df, which inflates variance and makes interpretation difficult. The **Lasso** was chosen as final model: nearly identical predictions with far fewer parameters.

### Main findings (Lasso)
- **Age / fertility**: younger women (start of fertility) have higher odds of using contraceptives than women at peak fertility; odds decrease in later phases. The subfertility coefficient is shrunk to zero.
- **Children**: compared with women without children, odds of use are lower for women with at least one child.
- **State**: Bihar and Uttar Pradesh, where sterilization targets were documented, have higher odds than Andhra Pradesh (reference).
- **Geography and religion**: lower odds in the South and North-East, and among Muslim women.
- **Education**: no particularly marked effect.

### Future work
Build a variable that explicitly flags the presence of coercive family-planning policies by state, to isolate their effect from the other socio-cultural determinants.

## Tech stack

R · `tidyverse` · `tidymodels` · `glmnet` · `mgcv` · `splines` · `ggplot2`

## References

1. Istituto Superiore di Sanità, *Linee guida 2024 PMA*.
2. Premala et al., *Fertility and Sterility*, 2014.
3. Human Rights Watch, *India: Target-Driven Sterilization Harming Women*, 2012.
4. The Hindu, *Indian women's ovaries age faster compared to Caucasian women: why?*

## Author

**Matteo Mascetti** (m.mascetti5@campus.unimib.it)
