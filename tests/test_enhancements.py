"""Run: python -m unittest discover -s tests -v"""
import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.bivariate_tree import BivariateDecisionTree, search_bivariate_split
from itertools import combinations


class EnhancementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = load_iris()
        cls.Xtrain, cls.Xtest, cls.ytrain, cls.ytest = train_test_split(
            d.data, d.target, stratify=d.target, test_size=.3, random_state=42,
        )
        cls.fixed = BivariateDecisionTree()
        cls.fixed.fit(cls.Xtrain, cls.ytrain)
        cls.adaptive = BivariateDecisionTree(adaptive_refinement=True)
        cls.adaptive.fit(cls.Xtrain, cls.ytrain)

    def test_default_is_original_fixed_grid(self):
        self.assertFalse(self.fixed.adaptive_refinement)
        self.assertTrue(np.array_equal(self.fixed.predict(self.Xtest),
            BivariateDecisionTree(adaptive_refinement=False).fit(
                self.Xtrain, self.ytrain).predict(self.Xtest)))

    def test_adaptive_can_only_improve_or_match_local_best_gini(self):
        pairs = list(combinations(range(4), 2))
        a = search_bivariate_split(self.Xtrain, self.ytrain, pairs, 18)
        b = search_bivariate_split(self.Xtrain, self.ytrain, pairs, 18,
                                   adaptive_refinement=True)
        self.assertGreaterEqual(b["gain"] + 1e-12, a["gain"])

    def test_explanation_matches_prediction(self):
        for x in self.Xtest[:10]:
            explanation = self.adaptive.explain_prediction(x)
            self.assertEqual(explanation["predicted_class"],
                             self.adaptive.predict(np.array([x]))[0])
            self.assertEqual(len(explanation["path"]),
                             sum(1 for _ in explanation["path"]))
            for step in explanation["path"]:
                self.assertEqual(step["direction"],
                                 "left" if step["score"] < 0 else "right")

    def test_explanation_readable_names_and_bad_inputs(self):
        d = load_iris()
        explanation = self.adaptive.explain_prediction(
            self.Xtest[0], d.feature_names, d.target_names,
        )
        self.assertIn(explanation["predicted_class_name"], d.target_names)
        with self.assertRaises(ValueError):
            self.adaptive.explain_prediction([2, 3])
        with self.assertRaises(ValueError):
            self.adaptive.explain_prediction([np.nan, 1, 2, 3])

    def test_invalid_refinement_parameters(self):
        with self.assertRaises(ValueError):
            BivariateDecisionTree(adaptive_refinement=True,
                                  refinement_step_degrees=0)
        with self.assertRaises(ValueError):
            BivariateDecisionTree(n_orientations=0)


if __name__ == "__main__":
    unittest.main()
