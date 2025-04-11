# import unittest
# from  


class TestWindowTemplateIterator(unittest.TestCase):
    def test_sorting(self):
        iterator = WindowTemplateIterator([3, 2, 1], 1)
        self.assertEqual(iterator.iterable, [1, 2, 3])
        
    def test_iter(self):
        iterator = WindowTemplateIterator([1, 2], 1)
        self.assertEqual(next(iterator), [1])
        self.assertEqual(next(iterator), [2])
        with self.assertRaises(StopIteration):
            next(iterator)
        
    def test_large_winlen(self):
        iterator = WindowTemplateIterator([1, 2, 3], 4)
        self.assertEqual(next(iterator), [1, 2, 3])
        
    def test_neg_winlen(self):
        with self.assertRaises(ValueError):
            WindowTemplateIterator([1, 2, 3], 0)
        with self.assertRaises(ValueError):
            WindowTemplateIterator([1, 2, 3], -1)
            
    def test_start_cord(self):
        iterator = WindowTemplateIterator([1, 2, 3], 1, 1)
        self.assertEqual(next(iterator), [2])
        
        
if __name__ == "__main__":
    unittest.main()