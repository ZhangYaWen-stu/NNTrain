import urllib.request
re = urllib.request.urlopen("https://www.python.org")
print(re.read().decode("utf-8"))
print(type(re))