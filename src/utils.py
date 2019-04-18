import os


def get_path_fname_ext(full_file_path):
    """Return path, filename without extension and extension
        ex: get_path_fname_ext('/Users/example/test.txt')
        ('/Users/example', 'test', '.txt')"""
    path, filename = os.path.split(full_file_path)
    fname, ext = os.path.splitext(filename)
    return path, fname, ext


def get_file_list(folder, extension=['.wav', '.aiff'], subdirectories=True):
    """ Returns list of all the files in the given folder
        recursively
    """
    file_list = []
    if subdirectories:
        for path, subdirs, files in os.walk(folder):
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in extension):
                    f = os.path.join(path, file)
                    file_list.append(f)
    else:
        for file in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, file)):
                if any(file.lower().endswith(ext.lower()) for ext in extension):
                    f = os.path.join(folder, file)
                    file_list.append(f)

    return file_list
