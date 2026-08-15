"""Patch fasttransform.Transform to make its `wrapped_enc` helper picklable on
Windows (`spawn` start-method) DataLoader workers.

Background: fasttransform.transform.Transform.__init__ defines a local closure
named `wrapped_enc` when `enc[0]` lacks a `__name__` attribute (Plum dispatch
requires the attribute). Local closures cannot be pickled, so any DataLoader
with `num_workers > 0` fails on Windows with::

    AttributeError: Can't pickle local object 'Transform.__init__.<locals>.wrapped_enc'

This patch replaces the closure with a module-level class
`_PicklableEncWrapper`. Instances carry `f` and `__name__` as attributes and
pickle cleanly because the class itself lives at module scope.

Re-run after every environment rebuild or fasttransform upgrade (mirrors the
existing patch_fastai_fakeloader.py convention).
"""
import fasttransform.transform as m

p = m.__file__
with open(p, encoding='utf-8') as f:
    c = f.read()

MARKER = "_PicklableEncWrapper"

if MARKER in c:
    print("Patch already applied to fasttransform.transform — no changes made.")
    raise SystemExit(0)

old_closure = (
    "                if not hasattr(enc[0],'__name__'): # Plum requires enc to have __name__ attr\n"
    "                    f = enc[0]\n"
    "                    def wrapped_enc(*args,**kwargs): return f(*args,**kwargs)\n"
    "                    wrapped_enc.__name__ = self._name\n"
    "                    enc[0] = wrapped_enc"
)

new_closure = (
    "                if not hasattr(enc[0],'__name__'): # Plum requires enc to have __name__ attr\n"
    "                    enc[0] = _PicklableEncWrapper(enc[0], self._name)  # PATCHED for Windows pickle"
)

if old_closure not in c:
    print("ERROR: did not find the expected wrapped_enc closure in fasttransform.transform.")
    print("       fasttransform may have changed upstream. Inspect the file manually:")
    print(f"       {p}")
    raise SystemExit(1)

helper_class = '''
# PATCHED for Windows pickle: module-level replacement for the local closure
# `wrapped_enc` originally defined inside Transform.__init__. Class instances are
# picklable (unlike local closures), so DataLoader workers can be spawned on
# Windows when num_workers > 0. Behaviour matches the original closure:
# instances forward all calls to the wrapped function `f` and carry a
# user-supplied `__name__` for Plum dispatch.
class _PicklableEncWrapper:
    def __init__(self, f, name):
        self.f = f
        self.__name__ = name
    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

'''

class_anchor = "class Transform(metaclass=_TfmMeta):"
if class_anchor not in c:
    print("ERROR: did not find Transform class definition in fasttransform.transform.")
    raise SystemExit(1)

c = c.replace(class_anchor, helper_class + class_anchor, 1)
c = c.replace(old_closure, new_closure, 1)

with open(p, "w", encoding='utf-8') as f:
    f.write(c)

print(f"Patched fasttransform.transform at {p}")
print("  - Inserted module-level _PicklableEncWrapper class")
print("  - Replaced local wrapped_enc closure with _PicklableEncWrapper(...) call")
