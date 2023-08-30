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
