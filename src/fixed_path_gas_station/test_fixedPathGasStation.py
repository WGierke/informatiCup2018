from unittest import TestCase
import pandas as pd
from fixed_path_gas_station import FixedPathGasStation


class TestCoordinate:
    def __init__(self, lat):
        self.lat = lat

    def dist(self, b):
        return b.lat - self.lat



class TestFixedPathGasStation(TestCase):
    def test_equidistant(self):
        path = pd.DataFrame({'cost': [1, 1],
                            'coords': [TestCoordinate(1), TestCoordinate(2)]})
        capacity = 1
        startfuel = 0
        res = FixedPathGasStation(path, capacity, startfuel,liter_per_100_km=100)
        self.assertEqual(res.price, 1)

    def test_it_only_gets_better(self):
        path = pd.DataFrame({'cost': [3, 2, 1],
                            'coords': [TestCoordinate(1), TestCoordinate(2), TestCoordinate(3)]})
        capacity = 3
        startfuel = 0
        res = FixedPathGasStation(path, capacity, startfuel,liter_per_100_km=100)
        print(res)
        self.assertEqual(res.prev, [0, 1, 2])
        self.assertEqual(res.fill_amount, [1, 1, 0])

    def test_insufficient_capacity(self):
        path = pd.DataFrame({'cost': [3, 2, 1],
                             'coords': [TestCoordinate(1), TestCoordinate(2), TestCoordinate(3)]})
        capacity = 0.5
        startfuel = 0
        with self.assertRaises(AssertionError):
            FixedPathGasStation(path, capacity, startfuel,liter_per_100_km=100)