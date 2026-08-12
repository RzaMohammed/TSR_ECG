import json

path = r'c:\TSR_ECG\TSRNet_Kaggle_Merge_Demo.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixed_train = False
fixed_test = False

for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))

    # Fix 1: Training cell — add --save_model 1
    if 'train.py' in src and '--save_path ckpt/' in src and '--save_model' not in src:
        cell['source'] = [
            "import os\n",
            "os.system('python train.py --data_path data/ --dims 12 --spec True --epochs 30 --batch_size 32 --save_path ckpt/ --save_model 1')\n",
        ]
        fixed_train = True
        print("Fixed: training cell — added --save_model 1")

    # Fix 2: Step 3 cell — robust checkpoint picker
    if "glob.glob('ckpt/TSRNet-*.pt')" in src or "glob.glob" in src and "TSRNet" in src:
        cell['source'] = [
            "import glob\n",
            "import os\n",
            "\n",
            "# Collect all epoch-numbered checkpoints (TSRNet-<epoch>.pt)\n",
            "def _epoch_key(p):\n",
            "    try:\n",
            "        return int(p.replace('\\\\', '/').split('/')[-1].split('-')[-1].split('.')[0])\n",
            "    except ValueError:\n",
            "        return -1  # fallback for 'latest'\n",
            "\n",
            "ckpts = sorted(glob.glob('ckpt/TSRNet-*.pt'), key=_epoch_key)\n",
            "\n",
            "# Filter to only epoch-numbered ones (exclude TSRNet-latest.pt as first choice)\n",
            "numbered = [c for c in ckpts if _epoch_key(c) >= 0]\n",
            "\n",
            "if numbered:\n",
            "    best_ckpt = numbered[-1]   # highest epoch = best saved AUC\n",
            "elif os.path.exists('ckpt/TSRNet-latest.pt'):\n",
            "    best_ckpt = 'ckpt/TSRNet-latest.pt'  # fallback to latest\n",
            "    print('WARNING: No epoch-numbered checkpoints found. Using TSRNet-latest.pt as fallback.')\n",
            "else:\n",
            "    raise FileNotFoundError(\n",
            "        'No checkpoint files found in ckpt/. '\n",
            "        'Make sure Step 2 (training) ran with --save_model 1 and completed at least 1 epoch.'\n",
            "    )\n",
            "\n",
            "print(f'Found {len(ckpts)} checkpoint(s). Using: {best_ckpt}')\n",
            "\n",
            "cmd = f'python test.py --data_path data/ --dims 12 --spec True --mask_loss True --load_model 1 --load_path \"{best_ckpt}\"'\n",
            "os.system(cmd)\n",
        ]
        fixed_test = True
        print("Fixed: Step 3 cell — robust checkpoint picker with --save_model 1 explanation")

if not fixed_train:
    print("WARNING: Training cell not found / already patched for --save_model 1")
if not fixed_test:
    print("WARNING: Step 3 cell not found")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook saved.")
