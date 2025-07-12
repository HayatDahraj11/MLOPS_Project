from src.data.data_loader import download_dataset, prepare_dataset

if __name__ == "__main__":
    print("Testing Kaggle dataset download...")
    download_dataset()
    print("Testing dataset preparation...")
    prepare_dataset()
    print("Test completed successfully!")