import re


class Lower:
    def __call__(self, unit: str):
        return unit.lower()


class Replace:
    def __init__(self, patterns):
        self.patterns = patterns

    def __call__(self, unit: str):
        for pattern, repl in self.patterns.items():
            unit = re.sub(pattern, repl, unit)
        return unit


class RemovePunc:

    def __init__(self, punctuations=None):
        if not punctuations:
            punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~«»…‚„“”‘’‹›–—•·¿¡`´¨¸×÷¤§©®™°±¼½¾‰¶'
        self.punctuations = punctuations

    def __call__(self, unit: str):
        return unit.translate(str.maketrans('', '', self.punctuations))
