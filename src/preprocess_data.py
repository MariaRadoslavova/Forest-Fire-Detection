import os
import shutil

# Step 2: Move the dataset into a folder called `data/`
os.makedirs('data/train', exist_ok=True)
for item in os.listdir('roboflow_data/train'):
    s = os.path.join('roboflow_data/train', item)
    d = os.path.join('data/train', item)
    shutil.move(s, d)

shutil.rmtree('roboflow_data')
