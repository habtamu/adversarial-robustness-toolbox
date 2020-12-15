%%writefile $script_folder/hardware_info.py

import platform
from multiprocessing import cpu_count
from socket import gethostname

def _generate_formatted_text(list_of_dicts):
    result = []
    for section in list_of_dicts:
        if section:
            text = ""
            longest = max(len(key) for key in section)
            for key, value in section.items():
                text += f"{key.ljust(longest)}: {value}\n"
            result.append(text)
    return "\n".join(result)    

def _get_pyversions():
    return {
        "Python implementation": platform.python_implementation(),
        "Python version": platform.python_version()
    }

def _get_sysinfo():
    return {
        "Compiler": platform.python_compiler(),
        "OS": platform.system(),
        "Release": platform.release(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "CPU cores": cpu_count(),
        "Architecture": platform.architecture()[0]
    }        


if __name__ == "__main__":
    output = []
    output.append(_get_pyversions())
    output.append(_get_sysinfo())
    output.append({"Hostname": gethostname()})
    print(_generate_formatted_text(output))    

#call 
#python test.py    