# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

excluded_modules = ['torch.distributions']  # add this
path = ['D:\\Workspace\\AutoStrike', 'D:\\Workspace\\AutoStrike',
        'D:\\Program Files\\JetBrains\\PyCharm\\plugins\\python\\helpers\\pycharm_display',
        'D:\\ProgramData\\Anaconda3\\python38.zip', 'D:\\ProgramData\\Anaconda3\\DLLs',
        'D:\\ProgramData\\Anaconda3\\lib',
        'D:\\ProgramData\\Anaconda3', 'D:\\ProgramData\\Anaconda3\\lib\\site-packages',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\natsort-7.1.1-py3.8.egg',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\timm-0.1.20-py3.8.egg',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\munkres-1.1.4-py3.8.egg',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\easydict-1.9-py3.8.egg',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\tensorboardx-2.2-py3.8.egg',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\visdom-0.1.8.9-py3.8.egg',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\terminaltables-3.1.0-py3.8.egg',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\websocket_client-0.59.0-py3.8.egg',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\torchfile-0.1.0-py3.8.egg',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\jsonpatch-1.32-py3.8.egg',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\jsonpointer-2.1-py3.8.egg',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\toolset-1.0-py3.8.egg',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\nms_module-1.0-py3.8-win-amd64.egg',
        'd:\\workspace\\hikvision',
        "D:\ProgramData\Anaconda3\lib\site-packages\cv2",
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\win32',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\win32\\lib',
        'D:\\ProgramData\\Anaconda3\\lib\\site-packages\\Pythonwin',
        'D:\\Program Files\\JetBrains\\PyCharm\\plugins\\python\\helpers\\pycharm_matplotlib_backend']
path = ["D:/Workspace/AutoStrike"] + path

a = Analysis(['auto_strike.py'],
             pathex=path,
             binaries=[],
             datas=[],
             hiddenimports=['pkg_resources.py2_warn'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=excluded_modules,
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='auto_strike',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='auto_strike')

#for d in a.datas:
#    if '_C.cp37-win_amd64.pyd' in d[0]:
#        a.datas.remove(d)
#        break
