import ctypes as ct

if __name__ == '__main__':
    dll = ct.windll.LoadLibrary(
        r"D:\Program Files\cudnn\bin/cudnn_cnn_infer64_8.dll")
    print(dll)
