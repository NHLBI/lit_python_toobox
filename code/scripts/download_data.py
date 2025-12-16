from tqdm.auto import tqdm
import requests
import pathlib
# A script inspired by the one located here:
# https://github.com/mikgroup/extreme_mri/blob/master/download_dataset.py

def download(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        with path.open('wb') as f:
            with tqdm(desc='Downloading ' + path.name, total=total_size,
                      unit='iB', unit_scale=True) as pbar:
                for data in r.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))



def download_data():
    coord_path = pathlib.Path('data/bcoord.npy')
    ksp_path = pathlib.Path('data/bksp.npy')
    download('https://zenodo.org/records/15802530/files/bcoord.npy', coord_path)
    download('https://zenodo.org/records/15802530/files/bksp.npy', ksp_path)
    

if __name__ == '__main__':
    download_data()
    print("Data downloaded successfully.")