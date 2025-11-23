# Simulation Results Report

## Experiment Summary

This simulation investigated how increasing the noise-to-tonal ratio (NTR) in vocal parameters affects the prediction accuracy of five machine learning algorithms (Naive Bayes, Decision Tree, Random Forest, KNN, and Logistic Regression) for Parkinson's Disease classification. The experiment systematically varied noise levels from baseline (1.0×) to 5× baseline to test the hypothesis that higher noise degrades prediction accuracy, with different algorithms showing varying degrees of robustness.

---

## Goal & Hypothesis

**Hypothesis:** Increasing the ratio of noise to tonal components in vocal parameters will decrease the prediction accuracy of machine learning algorithms for Parkinson's Disease classification, with the magnitude of accuracy reduction varying systematically across different algorithm types.

**Justification:** The ratio of noise to tonal components is a key indicator of voice quality degradation in PD-affected voices. Since vocal parameters influence prediction accuracy and different machine learning algorithms have varying noise tolerance properties, systematically manipulating NTR should reveal algorithm-specific robustness patterns.

---

## What We Planned to See

The expected outcomes included:

1. **Systematic accuracy decrease** for all five algorithms as noise-to-tonal ratio increases from 1.0× to 5.0× baseline
2. **Algorithm ranking by robustness** (most to least): Random Forest > KNN > Logistic Regression > Decision Tree > Naive Bayes
3. **Linear accuracy decline** for noise_multiplier 1.0-3.0, with potential plateau at higher levels
4. **Random Forest maintaining highest accuracy** across all noise levels due to ensemble averaging
5. **Expected accuracy reductions at 5.0× noise:**
   - Naive Bayes: 30-40%
   - Decision Tree: 25-35%
   - Logistic Regression: 20-30%
   - KNN: 15-25%
   - Random Forest: 10-20%
6. **Strong negative correlation** (r < -0.8) between noise_multiplier and accuracy
7. **Clear visual separation** between algorithm performance curves

---

## What We Actually Observed

### Baseline Performance (from STDOUT)

The simulation successfully completed and reported baseline accuracies:

- **Naive Bayes:** 0.475 (47.5%)
- **Decision Tree:** 0.644 (64.4%)
- **Random Forest:** 0.475 (47.5%)
- **KNN:** 0.492 (49.2%)
- **Logistic Regression:** 0.559 (55.9%)

### Critical Observations

1. **Unexpected baseline rankings:** Decision Tree achieved the highest baseline accuracy (64.4%), while Random Forest matched Naive Bayes at the lowest (47.5%). This contradicts the expected pattern where Random Forest should outperform Decision Tree.

2. **Random Forest underperformance:** Random Forest showing identical accuracy to Naive Bayes (47.5%) is highly unusual and suggests potential implementation issues (e.g., insufficient trees, poor hyperparameters, or data issues).

3. **Missing noise variation results:** The STDOUT only shows baseline accuracies. There is no evidence of:
   - Accuracy measurements at different noise_multiplier levels (1.5×, 2.0×, etc.)
   - Accuracy reduction calculations
   - Correlation coefficients
   - Statistical analysis results

4. **Artifact generated:** A single CSV file (`classification_data.csv`) was produced, but without examining its contents, we cannot determine if it contains the full results matrix across all noise levels.

5. **No visualization outputs:** The expected plots (line plots, bar charts, heatmaps, box plots, scatter plots) were not mentioned in the artifacts list.

---

## Did We Meet the Expected Outcome?

**Answer: NO**

**Reasoning:**

The simulation did not meet the expected outcomes for several critical reasons:

1. **Incomplete output:** Only baseline accuracies were reported in STDOUT. The core hypothesis—that accuracy decreases with increasing noise—cannot be validated without seeing results across the noise_multiplier range (1.0 to 5.0).

2. **Anomalous baseline results:** Random Forest performing at the same level as Naive Bayes (47.5%) and worse than Decision Tree (64.4%) contradicts fundamental machine learning principles. Random Forest, as an ensemble method, should typically outperform single Decision Trees, especially on noisy data.

3. **Missing critical analyses:** No correlation coefficients, accuracy reduction percentages, statistical significance tests, or algorithm robustness rankings were reported.

4. **No visualizations:** The absence of plots makes it impossible to verify the expected patterns of accuracy decline curves, algorithm separation, or linear vs. nonlinear relationships.

5. **Potential execution issue:** The simulation may have terminated after baseline calculation or encountered an error in the noise variation loop that was silently handled, preventing the full experimental procedure from completing.

While the simulation achieved "success" status technically, it failed to produce the evidence needed to evaluate the hypothesis.

---

## Key Notes About the Simulation

- **Baseline calculation completed successfully:** All five algorithms were trained and evaluated on the original dataset without errors

- **Random Forest configuration concern:** The identical performance of Random Forest and Naive Bayes strongly suggests:
  - Too few trees in the ensemble (possibly n_estimators=1)
  - Overfitting or underfitting due to poor hyperparameter choices
  - Data preprocessing issues affecting ensemble methods differently
  - Possible random seed issues causing degenerate behavior

- **Decision Tree outperformance:** Decision Tree achieving 64.4% accuracy (highest among all algorithms) is plausible but unexpected given ensemble methods should generally be more robust

- **Missing loop execution:** The noise variation loop (Steps 4-6 in the procedure) appears not to have executed or not to have printed results

- **Single artifact limitation:** Only one CSV file was generated instead of the expected multiple visualization files (PNG/PDF plots)

- **No error messages:** STDERR is empty, suggesting the code ran without exceptions, but the logic may have failed silently or output was not captured

---

## Next Steps / Recommendations

### Immediate Actions

1. **Inspect the CSV artifact:** Examine `classification_data.csv` to determine if it contains results across all noise levels or only baseline data. This will clarify whether the full experiment ran but failed to print, or stopped early.

2. **Debug Random Forest configuration:** Investigate the Random Forest implementation:
   - Verify `n_estimators` parameter (should be ≥100)
   - Check `max_depth`, `min_samples_split`, and other hyperparameters
   - Ensure proper random state initialization
   - Validate that the ensemble is actually aggregating multiple trees

3. **Add verbose logging:** Insert print statements at each major step:
   - After each noise_multiplier iteration
   - After each algorithm training/testing cycle
   - When storing results in the matrix
   - Before generating visualizations

4. **Verify loop execution:** Add explicit confirmation that the nested loops (noise levels × algorithms) are executing:
   ```python
   print(f"Processing noise_multiplier={noise_mult}, algorithm={algo_name}")
   ```

### Code Improvements

5. **Implement progressive output:** Print intermediate results after each noise level, not just at the end, to ensure visibility even if later steps fail

6. **Add visualization generation checks:** Wrap plotting code in try-except blocks with explicit error reporting to identify if visualization failures are occurring silently

7. **Validate data modifications:** After modifying NTR values, print summary statistics to confirm the noise multiplication is actually changing the dataset

8. **Save intermediate results:** Write results to CSV after each noise level to preserve partial results if the simulation terminates early

### Experimental Refinements

9. **Hyperparameter tuning:** Before running the noise variation experiment, perform grid search or cross-validation to find optimal hyperparameters for each algorithm, especially Random Forest

10. **Sanity check with synthetic data:** Create a simple synthetic dataset with known noise characteristics to validate that the noise injection mechanism works as intended

11. **Increase sample size:** If using real data, ensure sufficient samples (>500) for reliable accuracy measurements across train/test splits

12. **Add confidence intervals:** Run multiple iterations with different random seeds and report mean ± standard deviation for each accuracy measurement

### Hypothesis Refinement

13. **Consider alternative metrics:** In addition to accuracy, track precision, recall, F1-score, and AUC-ROC, which may be more informative for imbalanced PD datasets

14. **Test incremental noise steps:** Use finer granularity (0.1 increments) in the 1.0-2.0 range to better capture the initial degradation pattern

15. **Isolate NTR effects:** Run a control experiment varying only NTR without correlated changes to other features to isolate its specific impact