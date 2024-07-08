import unittest
from automl import MLSystem, prueba_kr, criterion_, nombre_, indicadora, label_tg, label_tg_inv
import numpy as np

class TestMLSystem(unittest.TestCase):
    def test_entire_work_flow(self):
        system = MLSystem()
        result = system.run_entire_work_flow()
        diferencia = np.abs(result["accuracy train"] - result["accuracy test"])
        self.assertTrue(result["success"],"The ML system workflow should have be completed...")
        self.assertGreater(result["accuracy train"],0.8,"The model accuracy train be above 0.8")
        self.assertGreater(result["accuracy test"],0.8,"The model accuracy test be above 0.8")
        self.assertLess(diferencia,2,"The model hasn't got overfitting")
        
if __name__ == "__main__":
    unittest.main()