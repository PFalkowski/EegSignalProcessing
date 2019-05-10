import unittest
import EegSignalProcessing as eeg

class Test_File(unittest.TestCase):
    def test_GetFileNameWithoutExtension(self):
        with self.assertRaises(ValueError):
            eeg.File("fileThatDoesNotExist.txt")

if __name__ == '__main__':
    unittest.main()

