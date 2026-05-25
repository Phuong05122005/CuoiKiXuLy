import os
import requests
import pandas as pd

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=t"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    
    # Check for token (for large files, though these are small)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk: 
                f.write(chunk)

splits = {
    "train": {
        "sentences": "1nzak5OkrheRV1ltOGCXkT671bmjODLhP",
        "sentiments": "1ye-gOZIBqXdKOoi_YxvpT6FeRNmViPPv",
        "topics": "14MuDtwMnNOcr4z_8KdpxprjbwaQ7lJ_C",
    },
    "validation": {
        "sentences": "1sMJSR3oRfPc3fe1gK-V3W5F24tov_517",
        "sentiments": "1GiY1AOp41dLXIIkgES4422AuDwmbUseL",
        "topics": "1DwLgDEaFWQe8mOd7EpF-xqMEbDLfdT-W",
    },
    "test": {
        "sentences": "1aNMOeZZbNwSRkjyCWAGtNCMa3YrshR-n",
        "sentiments": "1vkQS5gI0is4ACU58-AbWusnemw7KZNfO",
        "topics": "1_ArMpDguVsbUGl-xSMkTF_p5KpZrmpSB",
    }
}

def main():
    print("Downloading data manually from Google Drive...")
    os.makedirs("data_raw", exist_ok=True)
    os.makedirs("data_csv", exist_ok=True)
    
    for split_name, ids in splits.items():
        print(f"Processing {split_name}...")
        data_dict = {}
        for feature_name, file_id in ids.items():
            file_path = os.path.join("data_raw", f"{split_name}_{feature_name}.txt")
            print(f"  Downloading {feature_name}...")
            download_file_from_google_drive(file_id, file_path)
            
            with open(file_path, "r", encoding="utf-8") as f:
                data_dict[feature_name] = [line.strip() for line in f.readlines()]
                
        # Create a dataframe
        df = pd.DataFrame({
            "sentence": data_dict["sentences"],
            "sentiment": [int(x) for x in data_dict["sentiments"]],
            "topic": [int(x) for x in data_dict["topics"]],
        })
        
        csv_path = os.path.join("data_csv", f"{split_name}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"  Saved {split_name} to {csv_path} (Shape: {df.shape})")

    print("All data successfully downloaded and saved to data_csv/!")

if __name__ == "__main__":
    main()

