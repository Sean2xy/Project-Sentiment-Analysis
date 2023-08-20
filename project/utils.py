import pdb

import pandas as pd


def load_data(path):
    all_data = []
    with open(path, "r",encoding="utf-8") as f:
        data = f.readlines()[1:]
        for each_line in data:

            tmp=each_line.split(",")
            all_data.append([",".join(tmp[2:]), int(tmp[1])-1])

    return all_data
