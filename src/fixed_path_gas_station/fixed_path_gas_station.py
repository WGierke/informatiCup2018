from math import *
from heapq import heappush, heappop


class Coordinate:
    """ 
    Maintain and calculate with Coordinates
    """
    _ERD_RADIUS = 6378.388

    def __init__(self, lat, long):
        """
        Form a Coordinate.
        :param lat: latitude
        :param long: longitude
        """
        self.lat = radians(lat)
        self.long = radians(long)

    def dist(self, b):
        """
        Distance bewtween a and b as defined in the task description.
        :param b: Coordinate
        :return: distance
        """
        return self._ERD_RADIUS * acos(
            sin(self.lat) * sin(b.lat) + cos(self.lat) * cos(b.lat) * cos(b.long - self.long))

    def __iter__(self):
        return [degrees(self.lat),degrees(self.long)].__iter__()

class MinHeap:
    """
    MinHeap with deletion in O(log n).
    
    Maintains items of form (cost, index). ignore_up_to_index specifies the index until which items are deletedif they are seen at the top.
    """
    heap = []
    ignore_up_to_index = -1  # ignore nothing

    def push(self, item):
        # item[0] = cost, item[1] = index
        heappush(self.heap, item)

    def pop(self):
        self.pop_all_ignored()
        return heappop(self.heap)

    def top(self):
        self.pop_all_ignored()
        return self.heap[0]

    def ignore_inlcuding(self, index):
        self.ignore_up_to_index = index

    def pop_all_ignored(self):
        while len(self.heap) > 0 and self.heap[0][1] <= self.ignore_up_to_index:
            heappop(self.heap)

    def __str__(self):
        return str(self.heap)


class FixedPathGasStation:
    """
    Find an optimal solution for the fixed path gas station problem.
    
    Algorithm from:
    Samir Khuller, Azarakhsh Malekian, and Julián Mestre. 2011. To fill or not to fill: The gas station problem. 
    ACM Trans. Algorithms 7, 3, Article 36 (July 2011), 16 pages. DOI=http://dx.doi.org/10.1145/1978782.1978791 
    """

    def __init__(self, path, liter_capacity, start_fuel, liter_per_100_km=5.6):
        self.liter_capacity = liter_capacity
        self.liter_per_100_km = liter_per_100_km

        self.start_fuel = start_fuel
        self.path = path

        self.pre_compute()
        self.check_assumptions()
        self.compute()

    def check_assumptions(self):
        if len(list(filter(lambda dist: dist > self.km_capacity, self.segment_distance))) != 0:
            raise AssertionError('Capacity is not enough to drive from every station to next.')

    def d(self, a, b):
        if a == b:
            return 0
        return sum(self.segment_distance[a:b])

    def pre_compute(self):
        # pathlength
        self.stops = len(self.path)

        # distances along path
        self.segment_distance = [self.path['coords'][i].dist(self.path['coords'][i + 1]) for i in range(self.stops - 1)]
        self.segment_distance.append(0)

        self.km_capacity = self.lit2km(self.liter_capacity)

    def compute(self):
        self.compute_next_prev()
        self.compute_fill_commands()
        self.compute_fill_amount()
        self.compute_price()
        return self.price

    def km2lit(self,km):
        return (km / 100) * self.liter_per_100_km

    def lit2km(self,liter):
        return (liter / self.liter_per_100_km) * 100

    def compute_next_prev(self):
        n = self.stops
        prev = [-1] * n  # previous or same gas station with cheapest gas within capacity
        next = [-1] * n  # next gas station with cheapest gas within capacity

        heap = MinHeap()
        upper = 0
        lower = 0
        heap.push((self.path['cost'][0], 0))
        prev[0] = 0
        next[-1] = n - 1

        while upper < n - 1:
            # advance window to next filling station
            upper += 1

            # remove all stations too far away
            while lower < upper - 1 and lower < n - 2 and self.d(lower, upper) > self.km_capacity:
                heap.ignore_inlcuding(lower + 1)
                # and add their previous
                next[lower] = heap.top()[1]
                lower += 1

            # add new filling station and get prev for it
            heap.push((self.path['cost'][upper], upper))
            prev[upper] = heap.top()[1]

        assert upper == n - 1  # we have put the sliding window to the front

        # remove all remaining ones apart from last one (which has no next) from heap
        while lower < n - 2:
            heap.ignore_inlcuding(lower + 1)
            next[lower] = heap.top()[1]
            lower += 1

        break_points = [i for i in range(n - 1) if prev[i] == i]
        break_points.append(n - 1)

        self.prev = prev
        self.next = next
        self.break_points = break_points

    def compute_fill_commands(self):

        fill_command_km = [0] * self.stops

        def drive_to_next(i, k, fill):
            x = i
            d_x_k = self.d(x, k)
            while d_x_k > self.km_capacity:
                fill[x] = 'fill up'  # todo only fill up to k
                x = self.next[x]
                d_x_k = self.d(x, k)
            fill[x] = d_x_k
            return fill

        for i in range(len(self.break_points) - 1):
            fill_command_km = drive_to_next(self.break_points[i], self.break_points[i + 1], fill_command_km)
        self.fill_command_km = fill_command_km

    def compute_fill_amount(self):
        fuel = self.start_fuel
        fill_liters = len(self.path) * [0]
        for i, f in enumerate(self.fill_command_km):
            if f == 'fill up':
                fill_liters[i] = self.liter_capacity - fuel
            else:
                fill_liters[i] = max(0, self.km2lit(f) - fuel)
            fuel = fuel + fill_liters[i] - self.km2lit(self.segment_distance[i])
        self.fill_liters = fill_liters

    def compute_price(self):
        self.price = sum([cost * liters for cost, liters in zip(self.path['cost'], self.fill_liters)])

    def __str__(self):
        display_format = "{}: {}"
        to_be_printed = [('next', self.next), ('prev', self.prev), ('break_points', self.break_points),
                         ('price', self.price), ('fill_amount', self.fill_liters), ('fill_command in km', self.fill_command_km)]
        return 'Solution uses:\n' + '\n'.join([display_format.format(name, var) for name, var in to_be_printed])
