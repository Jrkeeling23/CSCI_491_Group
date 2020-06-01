import unittest
from data_processing import Data

data = Data()

class MyTestCase(unittest.TestCase):

    def test_stemming(self):
        """
        Test whether stemming works correctly
        :return: None
        """
        pre_stem = data.train_tweet[0]
        data.stem()
        post_stem = data.train_tweet[0]
        self.assertNotEqual(len(pre_stem), len(post_stem))
        self.assertNotEqual(pre_stem,post_stem, "STEM: Successful")

    def test_bigrams(self):
        "Test that bigrams is instantiated correctly"
        bigrams = data.bigrams(data.train_tweet)
        sample_frame = bigrams[0]
        self.assertTrue(type(sample_frame[0]) is tuple)
        self.assertEqual(len(sample_frame[0]), 2)


if __name__ == '__main__':
    unittest.main()
