import os
import urllib.request
import time

BASE_URL = "https://raw.githubusercontent.com/ShintaroMinami/PyDSSP/master/tests/testset/TS50"
TARGET_DIR = "tests/testset/TS50"

def download(url, path):
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def main():
    # Download list
    list_url = f"{BASE_URL}/list"
    list_path = os.path.join(TARGET_DIR, "list")
    if not download(list_url, list_path):
        return

    with open(list_path, 'r') as f:
        targets = [line.strip() for line in f if line.strip()]

    print(f"Found {len(targets)} targets.")

    for target in targets:
        # PDB
        pdb_url = f"{BASE_URL}/pdb/{target}.pdb"
        pdb_path = os.path.join(TARGET_DIR, "pdb", f"{target}.pdb")
        if not os.path.exists(pdb_path):
            download(pdb_url, pdb_path)

        # DSSP
        dssp_url = f"{BASE_URL}/dssp/{target}.dssp"
        dssp_path = os.path.join(TARGET_DIR, "dssp", f"{target}.dssp")
        if not os.path.exists(dssp_path):
            download(dssp_url, dssp_path)

        time.sleep(0.1) # Be nice

if __name__ == "__main__":
    main()
