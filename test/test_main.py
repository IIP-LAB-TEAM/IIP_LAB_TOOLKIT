import time
import unittest 
from toolkit_package.toolkit import multiprocessor, saveToFile
from toolkit_package.toolkit import Result_metrics, Loss
import torch
import torch.nn as nn
import os

def check(num):
    time.sleep(0.3)
    return num

class TestStringMethods(unittest.TestCase):

    def test_multiprocessor(self):
        num = 4
        num_list = [i for i in range(num)]
        result = multiprocessor(check, num_list, core = 4, return_value=True)
        self.assertEqual(len(result), num)

    def test_saveToFile(self):
        num_list = [i for i in range(30)]
        result = saveToFile(num_list, "test.csv", prefix="./toolkit_package/result/", time_stamp=True)
        self.assertEqual(result, True)

    
if __name__ == '__main__':
    unittest.main()

# python -m toolkit_package.test.test_main