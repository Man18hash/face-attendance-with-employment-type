FaceAttendance
A simple face recognition attendance application using Python and GUI (Tkinter), powered by FaceNet (facenet-pytorch).

Clone Repository
Clone the repo
git clone https://github.com/Man18hash/face-attendance-with-employment-type.git

Change into project directory
cd face-attendance-with-employment-type

Setup & Dependencies
Create and activate a virtual environment:
python -m venv venv
Windows: venv\Scripts\activate
macOS/Linux: source venv/bin/activate

Install required packages:
pip install opencv-python numpy Pillow torch facenet-pytorch pandas fpdf openpyxl tkcalendar

Running the Application
With virtualenv activated:
python biboy.py

Building Executable with PyInstaller
From within the project root (with venv active), run:
python -m PyInstaller --onefile --windowed --name FaceAttendance --collect-data facenet_pytorch biboy.py

This will generate a single-file GUI executable named FaceAttendance.exe (on Windows) in dist/.

Custom spec (biboy.spec)
Place this at the root as biboy.spec if you need to customize data inclusion:

-- mode: python ; coding: utf-8 --
import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

collect all the .pt files (and any other data) inside facenet_pytorch
datas = collect_data_files('facenet_pytorch')

also include your local dataset folder and attendance.csv
datas += [
('dataset', 'dataset'),
('attendance.csv', '.'),
]

a = Analysis(
['biboy.py'],
pathex=['.'],
binaries=[],
datas=datas,
hiddenimports=[
'torch', # ensure torch is picked up
'PIL._tkinter_finder', # ensure PIL+Tkinter support
],
hookspath=[],
runtime_hooks=[],
excludes=[],
noarchive=False,
)

pyz = PYZ(
a.pure,
a.zipped_data,
cipher=block_cipher,
)

exe = EXE(
pyz,
a.scripts,
a.binaries,
a.zipfiles,
a.datas,
[],
name='FaceAttendance',
debug=False,
bootloader_ignore_signals=False,
strip=False,
upx=True,
console=False, # GUI app
)
