import unittest
from data_processing import Data

class MyTestCase(unittest.TestCase):
    def test_stemming(self):
        data = Data()
        pre_stem = data.train_tweet[0]
        data.stem()
        post_stem = data.train_tweet[0]
        self.assertNotEqual(len(pre_stem), len(post_stem))
        self.assertNotEqual(pre_stem,post_stem, "YAY")


if __name__ == '__main__':
    unittest.main()
