import pandas as pd
import cv2 as cv
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['MEDV'] = boston.target
boston_df.head()