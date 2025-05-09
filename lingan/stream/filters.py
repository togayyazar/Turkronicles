from typing import List


class Length:
    def __init__(self, min_length=3, max_length=15):
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, unit: str):
        return self.min_length <= len(unit) <= self.max_length


class OnlyAlpha:
    def __call__(self, unit: str):
        return unit.isalpha()


class BlackList:
    def __init__(self, black_list: List[str]):
        self.black_list = black_list

    def __call__(self, unit: str):
        return unit not in self.black_list


class WhiteList:
    def __init__(self, white_list: List[str]):
        self.white_list = white_list

    def __call__(self, unit: str):
        return unit not in self.white_list
