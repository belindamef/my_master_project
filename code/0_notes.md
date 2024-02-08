# Potenetially usefull code snippets

## Create output log .txt-file

out_file = open(f"{dir_mgr.paths.this_val_out_dir}/out.txt",
                 'w', encoding="utf8")
sys.stdout = out_file

## Optional value with typehint

```Python
from typing import Optional

class MyClass:
    def __init__(self):
        self.instance_attribute: Optional[int] = None  # Default value and type hint

obj = MyClass()
print(obj.instance_attribute)  # Outputs: None

# Assign a value later
obj.instance_attribute = 42
print(obj.instance_attribute)  # Outputs: 42
```

## Potentially useful functions

```python
def extract_number(f):
    """A function to extract numbers from a filename given as string (?)"""
    s = re.findall("\d+$", f)
    return (int(s[0]) if s else -1, f)
```

```python
def find_most_recent_data_dir(val_results_paths: str) -> str:
    """A function to find the most recent directory if dir_path contains
    date"""
    all_dirs = glob.glob(os.path.join(val_results_paths, "*"))
    most_recent_data = max(all_dirs, key=extract_number)
    return most_recent_data
```

## Dimensions



|          | n      | m     | S          | O              | beta         | Phi                 | Phi (dict of csc)  | Omega^a          | Omega (csc)      |
|----------|-----   |-----  |---------   |---------       |---------     |-----------          | ----------------   |------------      |-----------       |
| 2x2, 2 h | 48     | 126   | (48, 4)    |  (126, 5)      |(48, 1)       |(48, 48, 5)          |                    | (48, 126)        |                  |
|    bytes |        |       |            |                |384           |11520                |  63352             | 48384            | 17304            |
| 3x3, 2 h | 648    |14810  | (648, 4)   |(14810, 5)      |(648, 1)      |(648, 648, 5)        |                    | (648, 14810)     |                  |
|    bytes |        |       |20736       |1184800         |5328          |2099520              |  55760             | 76775040         | 26849384         |
| 4x4, 2 h | 3840   |5111742| (3840, 4)  |((5111742, 17)) |(3840, 1)     |(3840, 3840, 5)      |                    | (3840, 5111742)  |                  |
|    bytes |        |       |122880      |695196912       |30720         |73728000             |                    | 19629089280      |                  |
| 4x4, 4 h | 3840   |5111742| (3840, 4)  |((5111742, 17)) |(3840, 1)     |(3840, 3840, 5)      |                    | (3840, 5111742)  |                  |
|    bytes |        |       |            |                |              |                     |                    |                  |                  |
| 5x5, 6 h |26565000|???    |(26565000,4)|(????,26)       |(26565000,1)  |(26565000,26565000,5)|                    |(26565000,???)    |                  |
|    bytes |        |       |            |                |              |                     |                    |                  |                  |

Anmerkung: vorherige nicht-sparse Werte für Phi und Omega unterschätzt tatsächliche Größe, da ich numpy attribute .nbytes
verwendet hatte, und jetzt sys.getsizeof().
sys.getsizeof: This function returns the size of the object in bytes, including all memory used by the object, including the memory overhead of the object itself and any contained objects.
.nbytes: This attribute, when applied to NumPy arrays, returns the total number of bytes consumed by the array's elements. It does not include the memory overhead of the array object itself.

Beispiel:

```sys.getsizeof(simulator.agent.stoch_matrices.beta_0)```

512

```asizeof.asizeof(simulator.agent.stoch_matrices.beta_0)```

528

```simulator.agent.stoch_matrices.beta_0.nbytes```

384
