import unittest
def add_a_b(a, b):
    return a + b
class AddTestCase(unittest.TestCase):
    def test_a_b_1(self):
        sum = add_a_b(1, 2)
        self.assertEqual(sum, 2)
    def test_a_b_2(self):
        sum = add_a_b(2, 3)
        self.assertEqual(sum, 4)
if __name__ == "__main__":
    unittest.main()
