import os
import sys
from unittest import TestCase

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from src.models.training import train


class TestTraining(TestCase):
    def test_train_on_complete_data(self):
        model, df_future = train(1920, up_to_days=0)
        assert len(df_future) == 0
