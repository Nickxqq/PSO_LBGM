# PSO_LBGM

The aim of this study is to identify new predictive biomarkers as well as potential therapeutic targets for psoriatic diseases (PsD). Using an MR-based analytical framework, we utilized data from nine large proteogenomics studies to explore causal associations between genetically predicted plasma proteins and PsD. We further validated their expression at the transcript level. Additionally, protein biomarkers were evaluated in a large cohort, and proof-of-concept was provided to investigate whether these protein biomarkers could potentially improve the prediction model for PsD. Finally, we used siRNA to silence gene expression and assess the therapeutic role of the most promising proteins.

### Predictive Model:
The project developed a clinical risk prediction model using the LightGBM algorithm, which inherently supports missing data, allowing us to analyze features without imputation. These features spanned demographic characteristics, physical measures, biological assays, and lifestyle data.

To identify the most important predictors, we built code to train 1,000 models and ranked features based on their contribution to PsD prediction. Through hierarchical clustering and a sequential forward selection process, we narrowed down the top ten clinical predictors. SHapley Additive exPlanations (SHAP) were used to interpret the contribution of these predictors to PsD risk.

### Key Findings:
- **Protein Screening:** We screened proteins and identified candidate proteins significantly associated with incident PsD.
- **Transcript Validation:** The mRNA expression of eight candidate proteins showed significant differences between psoriatic and non-psoriatic samples, as well as between lesional and non-lesional skin in PsD patients.
- **Cohort Analysis:** In a large cohort over 12 years of follow-up, plasma levels of some targets were associated with a higher risk of incident PsD per SD increase, respectively. Incorporating them significantly improved the model based on clinical predictors.
- **Therapeutic Insights:** Silencing our targets resulted in a significant reduction of psoriasis-like lesions and inflammatory markers in vivo.

### Conclusion:
Our study has uncovered numerous plasma proteins associated with PsD. Among these candidate proteins, some targets stand out as both predictive biomarkers and promising therapeutic targets. The incorporation of these biomarkers significantly improved the predictive performance of our LightGBM-based model.

### Project Flowchart:
![Project Flowchart](https://github.com/user-attachments/assets/a31b4fc1-2c72-4c2b-812d-7d5d0d4f3489)


