# Simulation Results Report

## Experiment Summary

This simulation tested whether novel peptide encoding methods incorporating supervised feature transformation (PCA or NLF) applied to physicochemical properties of amino acids would achieve at least 5% higher classification accuracy (measured by AUC) compared to traditional encoding methods. The experiment was designed to validate findings from a paper proposing two novel encoding approaches for peptide classification tasks across multiple biological domains (HIV-protease, T-cell epitopes, and HLA binding peptides).

## Goal & Hypothesis

**Hypothesis:** Peptide encoding methods that incorporate supervised feature transformation (PCA or NLF) applied to physicochemical properties of amino acids will achieve at least 5% higher classification accuracy (measured by AUC) compared to traditional encoding methods when tested on peptide classification tasks with sequence lengths between 8-15 amino acids.

**Justification:** The hypothesis is grounded in the paper's demonstration that novel encoding methods based on physicochemical properties and supervised feature transformation show superior performance compared to traditional methods. The paper provides a mathematical framework (Pp_20+k formula) for incorporating physicochemical properties into encodings and explicitly demonstrates improvements across multiple classification problems.

## What We Planned to See

The expected outcomes included:

1. **AUC Performance Ranges:** Novel methods achieving AUC values of 0.80-0.95, while traditional methods range from 0.70-0.85
2. **Minimum 5% Improvement:** Novel methods showing at least 5% higher AUC compared to traditional methods
3. **Consistency Across Problems:** Improvements observed across all three classification problems (HIV-protease, T-cell epitopes, HLA binding)
4. **Sequence Length Effects:** More pronounced advantages for intermediate sequence lengths (10-12 amino acids)
5. **Method Comparisons:** PCA-based encoding performing slightly better than NLF for smaller datasets
6. **Statistical Significance:** p < 0.05 for the improvement
7. **Clear Visual Separation:** Visualizations showing distinct performance curves with minimal overlap
8. **Feature Importance:** Identification of 5-8 key physicochemical properties driving performance gains

## What We Actually Observed

**STDOUT Output:**
```json
{"AUC": 0.85, "Improvement": 5.0}
```

**Artifacts Generated:**
- `results.json` - Contains detailed results data
- `dataset1_peptides.csv` - Peptide dataset used in the simulation

**Key Observations from Output:**

1. **AUC Value:** The simulation reported an AUC of 0.85, which falls within the expected range for novel methods (0.80-0.95)
2. **Improvement Metric:** The improvement is exactly 5.0%, which precisely meets the minimum threshold specified in the hypothesis
3. **Successful Execution:** Exit code 0 indicates the simulation completed without errors
4. **Limited Output:** The stdout provides only aggregate metrics without detailed breakdowns by sequence length, classification problem, or encoding method

## Did We Meet the Expected Outcome?

**Answer: PARTIALLY**

**Reasoning:**

The simulation successfully demonstrates the core claim of the hypothesis - that novel encoding methods achieve at least 5% improvement in AUC (exactly 5.0% was observed). The reported AUC of 0.85 falls within the expected range for novel methods. However, the output is significantly less comprehensive than anticipated:

**Met Expectations:**
- ✓ Achieved the minimum 5% improvement threshold
- ✓ AUC value (0.85) within expected range for novel methods
- ✓ Simulation executed successfully without errors

**Unmet/Unclear Expectations:**
- ✗ No breakdown by sequence length (8-15 amino acids)
- ✗ No comparison across different classification problems (HIV-protease, T-cell epitopes, HLA binding)
- ✗ No distinction between PCA vs NLF performance
- ✗ No statistical significance testing results (p-values)
- ✗ No visualization artifacts generated (plots, heatmaps, ROC curves)
- ✗ No feature importance analysis
- ✗ No comparison showing traditional method baseline AUC

The simulation appears to have executed only a subset of the planned procedure, providing aggregate results rather than the comprehensive analysis outlined in the simulation plan. While the core hypothesis is supported, the lack of detailed breakdowns prevents validation of the more nuanced expected outcomes.

## Key Notes About the Simulation

- **Minimal Output:** The simulation produced only two metrics (AUC and Improvement) rather than the comprehensive analysis planned across multiple dimensions
- **Exact Threshold Match:** The 5.0% improvement exactly matches the hypothesis threshold, which could indicate either a successful validation or potentially a simplified/truncated simulation
- **Missing Visualizations:** Despite the plan calling for multiple plots (line plots, bar charts, heatmaps, ROC curves, box plots), no image artifacts were generated
- **Dataset Generated:** The presence of `dataset1_peptides.csv` suggests data generation occurred, but only one dataset file was created rather than separate files for the three classification problems
- **Results File:** The `results.json` artifact likely contains more detailed data than what was printed to stdout and should be examined
- **No Error Messages:** Clean execution with no warnings or errors suggests the code ran as written, but may not have implemented all planned steps
- **Baseline Comparison Missing:** Without the traditional method AUC value, we cannot verify the absolute performance levels, only the relative improvement

## Next Steps / Recommendations

### Immediate Actions:

1. **Examine results.json:** Parse the detailed results file to determine if more comprehensive data was collected but not printed to stdout
2. **Verify Implementation Completeness:** Review the simulation code to confirm all 10 procedure steps were implemented, particularly:
   - Steps 7-10 (comparison, statistical testing, visualization, feature analysis)
   - Multiple classification problems and sequence lengths
3. **Add Detailed Logging:** Enhance stdout output to report:
   - Baseline AUC for traditional methods
   - AUC for each encoding method separately
   - Results broken down by sequence length and classification problem
   - Statistical test results (p-values, confidence intervals)

### Code Enhancements:

4. **Implement Visualization Generation:** Add code to create and save the planned plots (ROC curves, performance vs sequence length, heatmaps)
5. **Add Statistical Testing:** Implement the paired t-tests with multiple runs (n=30) as specified in Step 8
6. **Feature Importance Analysis:** Complete Step 10 to identify which physicochemical properties drive performance
7. **Parameter Sweep:** Ensure the simulation varies all specified parameters (sequence length 8-15, k-lag 1-5, all encoding methods, all classification problems)

### Validation Steps:

8. **Run Multiple Iterations:** Execute 30+ runs with different random seeds to assess result stability and compute confidence intervals
9. **Verify Expected Ranges:** Confirm that traditional methods achieve AUC in 0.70-0.85 range as expected
10. **Test Sequence Length Effects:** Specifically examine whether performance peaks at intermediate lengths (10-12 amino acids)
11. **Compare PCA vs NLF:** Separate analysis of the two supervised transformation methods

### Hypothesis Refinement:

12. **Consider Adjusting Threshold:** If 5% is consistently achieved, consider whether a higher threshold (7-10%) might be more representative of the paper's claims
13. **Add Secondary Metrics:** Include accuracy, precision, recall, and F1-score alongside AUC for more comprehensive evaluation