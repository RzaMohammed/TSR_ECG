import json
import os

nb_path = r"c:\TESTING PROJECT\TSRNet-main\TSRNet-main\TSRNet_Kaggle_Merge_Demo.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb.get("cells", [])):
    if cell["cell_type"] != "code":
        continue
        
    source = cell["source"]
    new_source = []
    modified = False
    
    for line in source:
        # Cell 3
        if line.startswith("%pip install"):
            new_source.extend([
                "import sys\n",
                "import subprocess\n",
                "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"wfdb\", \"heartpy\", \"PyWavelets\", \"tqdm\", \"scikit-learn\"])\n"
            ])
            modified = True
        # Cell 4
        elif line.startswith("!git clone"):
            new_source.extend([
                "import os\n",
                "import subprocess\n",
                "if not os.path.exists(\"TSR_ECG\"):\n",
                "    subprocess.run([\"git\", \"clone\", \"-q\", \"-b\", \"improvements\", \"https://github.com/SKYGOD07/TSR_ECG.git\"])\n"
            ])
            modified = True
        elif line.startswith("%cd"):
            new_source.append("os.chdir(\"TSR_ECG\")\n")
            modified = True
        elif line.startswith("!ls"):
            new_source.append("os.system(\"ls -la ckpt/\")\n")
            modified = True
        # Cell 11 and 14
        elif line.startswith("!python"):
            if "{best_ckpt}" in line:
                new_source.extend([
                    "cmd = f'python test.py --data_path data/ --dims 12 --spec True --mask_loss True --load_model 1 --load_path \"{best_ckpt}\"'\n",
                    "os.system(cmd)\n"
                ])
            else:
                cmd = line.replace("!python", "python").strip()
                new_source.append(f"import os\nos.system('{cmd}')\n")
            modified = True
        else:
            new_source.append(line)
            
    if modified:
        cell["source"] = new_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook fixed!")
