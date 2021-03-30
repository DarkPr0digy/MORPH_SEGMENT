class Symbols:
    """Class contains symbols that are consistent across all input files"""

    def __init__(self):
        """
        Constructor for class
        """
        self.arr = []
        self.arr.append([",", "_", "PUNCT", "_"])
        self.arr.append([".", "_", "PUNCT", "_"])
        self.arr.append(["[", "_", "PUNCT", "_"])
        self.arr.append(["]", "_", "PUNCT", "_"])
        self.arr.append(["(", "_", "PUNCT", "_"])
        self.arr.append([")", "_", "PUNCT", "_"])
        self.arr.append(["/", "_", "PUNCT", "_"])
        self.arr.append([":", "_", "PUNCT", "_"])
        self.arr.append(["©", "_", "PUNCT", "_"])
        self.arr.append([";", "_", "PUNCT", "_"])
        self.arr.append(["\'", "_", "PUNCT", "_"])
        self.arr.append(["\"", "_", "PUNCT", "_"])
        self.arr.append(["", "_", "PUNCT", "_"])
        self.arr.append([" ", "_", "PUNCT", "_"])
        self.arr.append(["-", "_", "PUNCT", "_"])
        self.arr.append(["!", "_", "PUNCT", "_"])
        self.arr.append(["?", "_", "PUNCT", "_"])
        self.arr.append(["*", "_", "PUNCT", "_"])
        self.arr.append(["&", "_", "PUNCT", "_"])
        self.arr.append(["...", "_", "PUNCT", "_"])
        self.arr.append(["\n", "_", "PUNCT", "_"])
        self.arr.append(["(iziqu)/", "_", "PUNCT", "_"])

    def inArr(self, symbol):
        """Method to check if a given character is in this alphabet of symbols"""
        for symbols in self.arr:
            if symbols[0] == symbol:
                return True
        return False
