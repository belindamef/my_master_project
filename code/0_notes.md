# Potenetially usefull code snippets

## Create output log .txt-file

out_file = open(f"{dir_mgr.paths.this_val_out_dir}/out.txt",
                 'w', encoding="utf8")
sys.stdout = out_file

## Optional value with typehint:

```python
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
