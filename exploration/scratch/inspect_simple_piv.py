from openpiv.piv import simple_piv
import inspect

print("simple_piv signature:", inspect.signature(simple_piv))
print("simple_piv docstring:", simple_piv.__doc__)
