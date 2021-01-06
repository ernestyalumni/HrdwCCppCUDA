# Creating and starting a virtual environment for Python 3

Create a directory for a virtual environment:

```
/HrdwCCppCUDA/Voltron$ python3 -m venv ./venv/
```

Activate it:
```
/HrdwCCppCUDA/Voltron$ source ./venv/bin/activate
```
You should see the prompt have a prefix `(venv)`.

Deactivate it:
```
deactivate
```

# Pip install (while in the virtual environment) requirements.

Go to `Voltron/Voltron` (you'll want the `requirements.txt` file accessible)

```
pip install -r requirements.txt
```

Running `pip freeze` **before** and after gives a good idea to the user of what `pip` packages have been installed.
