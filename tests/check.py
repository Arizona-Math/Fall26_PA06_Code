"""Small public smoke tests. These are examples, not exhaustive validation."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import bindings as nb
import modeling as model


class PublicChecks(unittest.TestCase):
    def test_negative_dominant(self):
        A = np.diag([-4., 2., 1.])
        lam, v, k, res, ok = nb.power(A, np.ones(3), 200, 1e-11)
        self.assertTrue(ok)
        self.assertAlmostEqual(lam, -4., places=9)
        self.assertLess(np.linalg.norm(A@v-lam*v), 1e-9)

    def test_budget(self):
        A = np.diag([3., 1.])
        lam, v, k, res, ok = nb.power(A, [1., 1.], 0, 1e-12)
        self.assertEqual(k, 0)
        self.assertFalse(ok)
        self.assertAlmostEqual(res, np.linalg.norm(A@v-lam*v))

    def test_deflation(self):
        A = np.array([[4., 2.], [0., 1.]])
        np.testing.assert_allclose(nb.deflate(A, 4., [1., 0.]), [[0., 2.], [0., 1.]])

    def test_rectangular_qr(self):
        A = np.array([[1., 0.], [1., 1.], [1., 2.]])
        Q, R = nb.qr(A[::-1])  # noncontiguous view
        np.testing.assert_allclose(Q@R, A[::-1], atol=1e-12)
        np.testing.assert_allclose(Q.T@Q, np.eye(2), atol=1e-12)

    def test_least_squares(self):
        A = np.array([[1., 0.], [1., 1.], [1., 2.]])
        np.testing.assert_allclose(nb.least_squares(A, [1., 2., 2.]), [7/6, 1/2], atol=1e-12)

    def test_design(self):
        A, b = model.build_design(np.arange(1., 9.), 2)
        self.assertEqual(A.shape, (6, 6))
        np.testing.assert_allclose(A[0], [1., 3., 2., 1., 0., 1.], atol=1e-14)
        np.testing.assert_allclose(b, np.arange(3., 9.))

    def test_recursive_forecast(self):
        c = np.array([0., 0., .5, .25, 0., 0.])
        np.testing.assert_allclose(model.forecast([2., 4.], c, 2, 3), [2.5, 2.25, 1.75])


if __name__ == "__main__":
    unittest.main()
