import lvpyio
print("lvpyio contents:", dir(lvpyio))
for attr in dir(lvpyio):
    item = getattr(lvpyio, attr)
    if callable(item):
        print(f"Callable: {attr}, doc: {item.__doc__}")
