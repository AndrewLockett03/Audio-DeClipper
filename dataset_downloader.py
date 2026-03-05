import kagglehub


if __name__ == "__main__":
    dataset_path = kagglehub.dataset_download(
        "mozillaorg/common-voice"
    )

    print(dataset_path)
