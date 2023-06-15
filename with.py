class cat:
    def __enter__(self):
        print("__enter__")
        return self
    def __exit__(self,type,value,trace):
        print("__exit__")
        print(type)
        print(value)
        print(trace)
        return True
    def name(self,name):
        print(name/0)

def get_cat():
    return cat()
if __name__ == "__main__":
    #with get_cat() as c:
     #   c.name(12)
    #print(c.name(123))

    t=get_cat()
    print(t.name(12))
