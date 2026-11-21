# Extensions to the simplified Bivariate Decision Tree project

The original GitHub project implemented a **simplified** bivariate CART-like
algorithm from the KDD 2024 paper. This extension does **not** claim to
implement the paper's full TAO algorithm or establish a new research result.

## 1. Adaptive angle refinement (algorithmic enhancement)

The original fixed-angle model evaluates 18 orientations from 0 to 180 degrees
for each feature pair. The enhanced model:

1. Evaluates those same 18 coarse orientations.
2. Ranks the orientations for each feature pair using Gini impurity gain.
3. Selects the top 2 orientations for each pair.
4. Evaluates extra nearby angles within ±8°, at 2° increments, skipping
   duplicate coarse candidates.
5. Retains the candidate with the best Gini gain; still compares it with
   the univariate fallback. Ties still favor the univariate rule.

The extra search cannot worsen **the best local split Gini gain** on the same
node dataset, because the original candidates are all retained. It CAN make
whole-tree test accuracy worse, because training gain is not generalization.
It also increases training cost; experimental results are required.

Enable it with:

```python
from src.bivariate_tree import BivariateDecisionTree
model = BivariateDecisionTree(adaptive_refinement=True,
                             refinement_top_k=2,
                             refinement_window_degrees=8,
                             refinement_step_degrees=2)
model.fit(X_train, y_train)
```

`adaptive_refinement=False` is the default and preserves the original model.

## 2. Explainable predictions (interpretability enhancement)

```python
explanation = model.explain_prediction(
    X_test[0], feature_names=iris.feature_names,
    class_names=iris.target_names,
)
model.print_prediction_explanation(
    X_test[0], feature_names=iris.feature_names,
    class_names=iris.target_names,
)
```

For each decision node the explanation provides the features, coefficients,
threshold/bias, evaluated numerical score, branch taken and split-search stage.
It also reports the predicted class and leaf's training class counts. This is
**decision-path transparency**, not a causal explanation or proof of why the
label is correct.

## 3. Run evaluations

From the **repository root**:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py --quick
python main.py --enhanced
```

`python main.py` still runs the original experiment and preserves its original
outputs. `python main.py --enhanced` compares:

- Matched-depth sklearn DecisionTreeClassifier (depth 4, min split 5)
- Original fixed-angle bivariate model
- New adaptive bivariate model

on the same stratified Iris split and five-fold cross-validation folds.

**New files written (old files are not overwritten):**

- `results/adaptive_comparison.csv`
- `results/adaptive_prediction_explanation.json`
- `results/adaptive_cv_folds.csv`
- `results/adaptive_cv_summary.csv`
- `plots/adaptive_accuracy_comparison.png`
- `results/adaptive_synthetic_demo.csv`
- `plots/adaptive_synthetic_boundaries.png`

The separate synthetic demonstration creates 180 artificial two-feature
samples with a 27° diagonal decision pattern and measures whether finer
angles can improve a single-node tree. This is illustrative evidence of
functionality, **not** evidence that the enhanced model always improves
accuracy or that the original KDD paper reported these numbers.

Training times are machine-dependent. The baseline here uses matched stopping
hyperparameters for improved fairness, while the original comparison in
`src/experiment.py` remains intact and uses the unrestricted sklearn tree.

## 4. Points to explain to the supervisor

- **Original paper idea:** two-feature oblique decision rules; paper also
  describes optimization procedures not reproduced here.
- **Original student code:** coarse 18-angle grid and local Gini gain; ordinary
  univariate fallback.
- **Our added algorithmic feature:** per-feature-pair adaptive coarse-to-fine
  angle search.
- **Our added interpretability feature:** explanation of any sample's actual
  tree traversal with numeric scores.
- **Our additional validation:** experiment comparing all three models on the
  same split and CV folds, with separate generated output files.
- **Limitations:** finer search is slower; gains in local impurity do not imply
  better held-out accuracy; small Iris dataset alone is insufficient for a
  broad effectiveness claim.
