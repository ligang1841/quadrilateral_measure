# -*- mode: python ; coding: utf-8 -*-

#block_cipher = pyi_crypto.PyiBlockCipher(key='ms999(ID')
block_cipher = None

a = Analysis(['measure.py'],
             pathex=['C:\\Users\\fourai\\code\\quadrilateral_measure'],
             binaries=[('m_config.txt','.')],
             datas=[],
             hiddenimports=[ 'pkg_resources.py2_warn'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

#avoid warning
#for d in a.datas:
#    if '_AES.cp38-win_amd64.pyd' in d[0]:
#        a.datas.remove(d)
#        break

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='measure',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
