import os
import glob

# Search for pkl files in your project
pkl_files = glob.glob('**/*.pkl', recursive=True)
print("Found PKL files:", pkl_files)

# Search in common directories
possible_locations = [
    '.', 
    './models', 
    './data',
    '../',
    os.path.expanduser('~')
]

for location in possible_locations:
    if os.path.exists(os.path.join(location, 'internship_recommender.pkl')):
        print(f"Found at: {os.path.join(location, 'internship_recommender.pkl')}")