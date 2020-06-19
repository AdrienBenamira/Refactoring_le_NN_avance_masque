import pandas as pd
from sympy.logic import SOPform
from sympy import symbols
import numpy as np

"""w, x, y, z = symbols('DL[i-1], DV[i-1],  V0[i-1], V1[i-1]')
minterms = [[0, 0, 0, 1], [0, 0, 1, 1],
            [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]]
exp =SOPform([w, x, y, z], minterms)
print(exp)"""



df = pd.read_csv("table_of_tructh_0922.csv")
df_name = pd.read_csv("table_of_tructh_0922_name.csv")

df = df.rename(columns={"Unnamed: 0": "Key"})
df = df.rename(columns={str(i): "Filter_" + str(i) for i in range(64)})

df_name = df_name.rename(columns={"Unnamed: 0": "Key",
                                  "0": "DL[i-1]",
                                  "1": "DV[i-1]",
                                  "2": "V0[i-1]",
                                  "3": "V1[i-1]",
                                  "4": "DL[i]",
                                  "5": "DV[i]",
                                  "6": "V0[i]",
                                  "7": "V1[i]",
                                  "8": "DL[i+1]",
                                  "9": "DV[i+1]",
                                  "10": "V0[i+1]",
                                  "11": "V1[i+1]",

                                  })

print(df.head(5))
print(df_name.head(5))
df_m = pd.merge(df_name, df, on='Key')
print (df_m.head(5))
del df_m['Key']
print (df_m.head(5))

for index_f in range(64):
    print(index_f)
    index_intere = df_m.index[df_m['Filter_'+str(index_f)] == 1].tolist()
    if len(index_intere) ==0:
        print("Empty")
    else:
        condtion_filter = []
        for col in ["DL[i-1]", "V0[i-1]", "V1[i-1]", "DL[i]", "V0[i]", "V1[i]", "DL[i+1]", "V0[i+1]", "V1[i+1]"]:
            s = df_m[col].values
            condtion_filter.append(s[index_intere])
        condtion_filter2 = np.array(condtion_filter).transpose()
        condtion_filter3 = [x.tolist() for x in condtion_filter2]
        assert len(condtion_filter3) == len(index_intere)
        assert len(condtion_filter3[0]) == 9
        w1, x1, y1, w2, x2, y2, w3, x3, y3 = symbols('DL[i-1], V0[i-1], V1[i-1], DL[i], V0[i], V1[i], DL[i+1], V0[i+1], V1[i+1]')
        minterms = condtion_filter3
        exp =SOPform([w1, x1, y1, w2, x2, y2, w3, x3, y3], minterms)
        print(exp)