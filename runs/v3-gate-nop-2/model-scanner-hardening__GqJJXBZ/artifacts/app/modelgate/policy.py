"""What the scanner looks for.

Pickle: a blocklist of callables that execute code, start processes, touch
files or open connections when unpickled.  TensorFlow: the graph ops that
read or write files.  Keras: Lambda layers, which carry serialized bytecode.
"""

# module.name -> finding kind
PICKLE_BLOCKLIST = {
    "os.system": "process",
    "os.popen": "process",
    "os.execv": "process",
    "os.execvp": "process",
    "os.spawnl": "process",
    "subprocess.Popen": "process",
    "subprocess.call": "process",
    "subprocess.run": "process",
    "subprocess.check_output": "process",
    "subprocess.check_call": "process",
    "builtins.eval": "code-execution",
    "builtins.exec": "code-execution",
    "builtins.compile": "code-execution",
    "builtins.__import__": "code-execution",
    "builtins.open": "file-access",
    "shutil.rmtree": "file-access",
    "shutil.copyfile": "file-access",
    "socket.socket": "network",
    "socket.create_connection": "network",
    "urllib.request.urlopen": "network",
    "webbrowser.open": "network",
    "pty.spawn": "process",
}

# Strings that should never appear inside a legitimate model file.  Pickles
# are scanned for them as well as for the globals above.
SUSPICIOUS_STRINGS = ("os.system", "subprocess", "eval(", "exec(")

# TensorFlow ops that read or write arbitrary files.
TF_FILE_OPS = {"ReadFile": "file-access", "WriteFile": "file-access"}
