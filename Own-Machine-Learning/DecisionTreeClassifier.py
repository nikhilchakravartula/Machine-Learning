import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gini_index(groups,class_values):
    gini_value=0
    for group in groups:
        for class_value in class_values:
            to_be_added= 