import subprocess
import sys

required_libraries = ['opencv-python', 'numpy', 'matplotlib']

def install_libraries():
    for lib in required_libraries:
        try:
            __import__(lib)
        except ImportError:
            print(f'{lib} not found, installing...')
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

install_libraries()