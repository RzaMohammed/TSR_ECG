import json

path = r'c:\TSR_ECG\TSRNet_Kaggle_Merge_Demo.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixed = False
for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))

    # Target the Step 3 cell
    if "glob.glob('ckpt/TSRNet-*.pt')" in src or ("glob.glob" in src and "TSRNet" in src and "best_ckpt" in src):
        cell['source'] = [
            "import glob\n",
            "import os\n",
            "\n",
            "def _epoch_key(p):\n",
            "    try:\n",
            "        return int(p.replace('\\\\', '/').split('/')[-1].split('-')[-1].split('.')[0])\n",
            "    except ValueError:\n",
            "        return -1\n",
            "\n",
            "def find_best_ckpt():\n",
            "    ckpts = sorted(glob.glob('ckpt/TSRNet-*.pt'), key=_epoch_key)\n",
            "    numbered = [c for c in ckpts if _epoch_key(c) >= 0]\n",
            "    if numbered:\n",
            "        return numbered[-1]\n",
            "    if os.path.exists('ckpt/TSRNet-latest.pt'):\n",
            "        return 'ckpt/TSRNet-latest.pt'\n",
            "    return None\n",
            "\n",
            "# Auto-train if no checkpoint exists yet\n",
            "if find_best_ckpt() is None:\n",
            "    print('No checkpoint found — running training first (this may take a while)...')\n",
            "    os.makedirs('ckpt', exist_ok=True)\n",
            "    ret = os.system('python train.py --data_path data/ --dims 12 --spec True --epochs 30 --batch_size 32 --save_path ckpt/ --save_model 1')\n",
            "    if ret != 0:\n",
            "        raise RuntimeError('Training failed (non-zero exit code). Check the output above for errors.')\n",
            "\n",
            "best_ckpt = find_best_ckpt()\n",
            "if best_ckpt is None:\n",
            "    raise FileNotFoundError(\n",
            "        'Training completed but no checkpoint was saved. '\n",
            "        'This can happen if AUC never improved and TSRNet-latest.pt was not written. '\n",
            "        'Check that data/train.npy, data/test.npy, and data/label.npy exist.'\n",
            "    )\n",
            "\n",
            "print(f'Using checkpoint: {best_ckpt}')\n",
            "cmd = f'python test.py --data_path data/ --dims 12 --spec True --mask_loss True --load_model 1 --load_path \"{best_ckpt}\"'\n",
            "os.system(cmd)\n",
        ]
        fixed = True
        print("Step 3 cell updated: auto-trains if no checkpoint found.")
        break

if not fixed:
    print("ERROR: Step 3 cell not found.")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook saved.")
