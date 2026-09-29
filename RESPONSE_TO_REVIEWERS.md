# Response to Reviewers

**Manuscript title:** Explainable machine learning and correlation-based network analysis of self-reported adolescent suicidality in South Korea: a repeated cross-sectional study, 2015–2024  
**Submission ID:** 03ec0a51-7c19-4748-ac24-79770fc64bf2  
**Journal:** BMC Public Health  
**Handling Editor:** Lay San Too

Dear Editor and Reviewers,

We sincerely thank the Handling Editor and both reviewers for their careful and constructive assessment of our manuscript. We have revised the manuscript and supplementary material substantially. In particular, we (1) added threshold-sensitivity analyses using the default 0.50 threshold, Youden’s J, and the threshold maximizing the positive-class F1-score; (2) clarified the descriptive and marginal nature of the Pearson correlation network and revised its visualization so that retained edge magnitudes are represented continuously; (3) expanded the limitations and future directions to discuss graphical models appropriate for mixed data; (4) added comparison with prior KYRBS machine-learning studies of adolescent suicidality; (5) clarified how temporal heterogeneity and structural wave-specific predictor availability were handled; and (6) aligned the Data Availability statement and supplementary legends with the journal’s editorial requirements. We also reviewed the manuscript to avoid overstating predictive or clinical implications.

Below we respond point by point. Reviewer comments are reproduced in bold, followed by our responses.

---

## Editorial comments

### Editorial comment 1

**When submitting your revised manuscript, please select ‘yes’ under the data availability declaration and input your data availability statement in the appearing textbox. Kindly ensure to provide the same data availability statement on the submission system and in the manuscript.**

**Response:** Thank you. We have ensured that the revised manuscript contains the Data Availability statement and will enter the identical statement in the submission system while selecting “yes” for the data availability declaration. The study uses Korea Youth Risk Behavior Survey (KYRBS) data obtained from the Korea Disease Control and Prevention Agency (KDCA); raw microdata are not redistributed in our public code repository. We have also made the analysis code available through the project GitHub repository to improve reproducibility.

### Editorial comment 2

**We have noticed that legends for supplementary figures and tables has been provided in the main manuscript. Kindly remove it from the manuscript and provide it in the supplementary file itself near to the supplementary figures and tables and not as a separate upload.**

**Response:** Thank you. We removed the supplementary figure and table legends from the main manuscript and placed the relevant titles/legends directly with the corresponding supplementary tables and figures in the supplementary file.

---

## Reviewer 1

We thank Reviewer 1 for the constructive suggestions, which helped us improve both the practical interpretation of the prediction models and the methodological framing of the network analysis.

### Comment 1

**While the authors already note this limitation, providing a supplementary analysis or table illustrating model performance under optimal decision thresholds (e.g., chosen via Youden’s Index, or maximizing minority-class F1-score) would add substantial practical and clinical insight for triage/screening exploration.**

**Response:** We agree and have added a threshold-sensitivity analysis for each suicidality outcome. In addition to the prespecified/default probability threshold of 0.50, we evaluated (1) the threshold maximizing Youden’s J and (2) the threshold maximizing the positive-class (minority-class) F1-score. The analysis reports sensitivity, specificity, positive predictive value, negative predictive value, positive-class F1-score, balanced accuracy, and accuracy and is presented in Supplementary Table S12.

We emphasize that these alternative thresholds are exploratory rather than clinically validated cutoffs. Because threshold selection and evaluation were performed using the same held-out prediction set, the resulting operating points should not be interpreted as externally validated screening or triage thresholds. We therefore use this analysis to demonstrate the trade-off between sensitivity and false-positive burden under different operating points rather than to recommend a clinical threshold. The Methods, Results, and Discussion/limitations were revised accordingly.

### Comment 2

**The manuscript already acknowledges this as a limitation, but it would be beneficial to explicitly mention in the Future Directions section that alternative graphical models suited for mixed data (such as Graphical Gaussian Models [GGM], Polychoric/Tetrachoric correlations, or Ising models) could be explored to validate the structural stability of the network.**

**Response:** We agree. We expanded the limitations/future-directions discussion to state explicitly that the present Pearson correlation network is a descriptive representation of marginal cross-sectional associations and that future work should assess structural stability using graphical approaches better suited to mixed or categorical data. We now specifically mention Graphical Gaussian Models, polychoric/tetrachoric correlation approaches, and Ising models as potential alternatives. This addition also clarifies that the current network should not be interpreted as a conditional-independence, causal, or temporal network.

### Comment 3

**Machine learning studies on suicide-related outcomes using the KYRBS have already been published. Therefore, the Discussion should include a comparative analysis with these previous studies.**

**Response:** Thank you. We expanded the Discussion to compare our study with the three suggested KYRBS-based machine-learning studies. Kim et al. developed models for suicidal thinking using KYRBS data and external validation cohorts from the United States and Norway, with XGBoost showing strong discrimination. Lee et al. focused specifically on suicide attempts among adolescents with allergic rhinitis and used KYRBS for model development with KNHANES as an external validation dataset. Wang et al. compared multiple machine-learning algorithms for adolescent suicide risk using national Korean survey data and combined model interpretation with feature/interaction analyses. We now distinguish our contribution from these studies by emphasizing the repeated 2015–2024 framework, separate modeling of suicidal ideation, planning, and attempts, survey-weighted and PSU-aware validation, temporal robustness assessment, and integration of explainable prediction with a descriptive correlation-network analysis. We have used the suggested references where they directly improve the comparison rather than treating citation of any particular reference as mandatory.

---

## Reviewer 2

We thank Reviewer 2 for the careful methodological assessment. We agree that the original manuscript needed clearer justification and interpretation of the correlation network and a more explicit description of temporal heterogeneity across survey waves.

### Comment 1

**The authors constructed a correlation-based network using pairwise Pearson correlation coefficients. However, such correlations represent marginal associations and do not account for other variables that may contribute to the observed relationship between two nodes. The authors should better justify choosing a correlation-based network over alternative approaches.**

**Response:** We agree. We have revised the Methods and Discussion to make the purpose and limitations of the network analysis explicit. The Pearson correlation network was selected as a descriptive complement to the predictive modeling rather than as a conditional-dependence or causal graphical model. Its purpose is to provide a transparent, reproducible overview of pairwise co-occurrence patterns among suicidality outcomes and measured psychosocial/behavioral variables in the repeated cross-sectional survey data. We now explicitly state that an edge represents a marginal association and may reflect direct association, shared covariation with other measured variables, or unmeasured factors. Consequently, we do not interpret network edges as independent effects, causal pathways, or temporal relationships.

We also strengthened the limitations and future-directions section by noting that Graphical Gaussian Models, polychoric/tetrachoric correlation approaches, and Ising models could be used in future work to examine whether the observed structure is robust to methods that better accommodate conditional dependence and mixed/categorical variables. These revisions are intended to ensure that the correlation network is interpreted only within its descriptive scope.

### Comment 2

**The construction and visualization of the correlation network depend on prespecified thresholds for the magnitude of the Pearson correlation coefficient. Although the authors appropriately state that these thresholds do not represent statistical significance, the rationale for selecting these particular cutoffs remains unclear. Alternatively, displaying the magnitude of correlations continuously, for example through edge width or another visual property, may avoid imposing arbitrary categories on the associations.**

**Response:** We agree and revised the network visualization. Rather than assigning retained correlations to discrete magnitude categories, edge width is now scaled continuously according to the absolute correlation magnitude (|r|). Positive and negative correlations remain visually distinguishable, while line width directly reflects correlation strength. A minimal |r| < 0.10 filter is retained only to reduce visual clutter and improve readability of the network; it is not presented as a statistical-significance threshold or a substantive boundary between meaningful and non-meaningful associations. The Methods and Figure 4 legend have been revised to make this distinction explicit. This change reduces reliance on arbitrary categorical cutoffs while preserving a readable descriptive network.

### Comment 3

**It is unclear how the primary models accounted for temporal variation across the ten survey waves. Although leave-one-year-out validation evaluates temporal robustness, survey year does not appear to have been included as a predictor or otherwise explicitly modeled. Given secular changes in adolescent suicidality and predictor distributions over 2015–2024, as well as substantial survey-year-specific structural non-availability of several predictors, the authors should clarify how they addressed temporal heterogeneity.**

**Response:** Thank you for highlighting this important point. We have clarified the temporal design in the Methods and limitations. Survey year was intentionally not included as a predictor in the primary models. The aim was to estimate associations/predictive patterns based on participant-level characteristics rather than allow the model to use calendar year itself as a proxy for secular trends. Temporal heterogeneity was instead evaluated through leave-one-year-out validation, in which each survey year was withheld in turn to assess generalization to an unseen wave.

We also clarified our handling of structural predictor non-availability across waves. The primary 2015–2024 models were restricted to predictors available and harmonizable across the full study period. Variables that were introduced only in later waves were not treated as ordinary participant-level missing values in the primary analysis. In particular, loneliness was not included in the full-period primary predictor set because it was structurally unavailable in earlier waves; it was considered separately in a 2020–2024 sensitivity analysis. Thus, the primary model avoids imputing a variable across years in which it was not collected.

We now state more explicitly that leave-one-year-out validation evaluates temporal robustness but does not fully model secular change. Residual heterogeneity across waves may remain because the study is repeated cross-sectional, predictor distributions and outcome prevalence can change over time, and survey instruments may evolve. We therefore avoid interpreting temporal validation as evidence that the model captures or explains time trends and identify explicit modeling of temporal drift and wave-specific measurement changes as an area for future work.

---

## Summary of major revisions

The revised submission therefore includes: (1) threshold-sensitivity analysis with default, Youden-J, and positive-class-F1 operating points (Supplementary Table S12); (2) revised Figure 4 with continuous edge-width scaling; (3) clearer justification and limitations of the descriptive Pearson correlation network; (4) future directions covering GGM, polychoric/tetrachoric correlations, and Ising models; (5) expanded comparison with prior KYRBS machine-learning studies; (6) explicit clarification of temporal heterogeneity, leave-one-year-out validation, and structural wave-specific variable availability; (7) harmonized Data Availability reporting; and (8) relocation of supplementary legends from the main manuscript to the supplementary file.

We appreciate the Editor’s and Reviewers’ comments, which have substantially improved the clarity, reproducibility, and methodological framing of the manuscript.