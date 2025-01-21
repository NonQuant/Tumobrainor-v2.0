import os
import pickle
import cv2


labels = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "notumor": 3,
}


def load_testing_data(data_path):
    testing_data = []

    # adding samples to testing data list
    img_count = 0
    for i, category in enumerate(os.listdir(data_path)):
        if category not in labels.keys():
            continue
        category_path = os.path.join(data_path, category)
        for filename in os.listdir(category_path):
            filepath = os.path.join(category_path, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (512, 512))
            label = labels[category]
            testing_data.append([img, label])

            img_count += 1

    print(f"TESTING DATA: Loaded: {len(testing_data)} files")
    print(f"shape: {testing_data[0][0].shape} label: {testing_data[0][1]}")

    return testing_data


def load_training_data(data_path):
    training_data = []

    # adding samples to training data list
    img_count = 0
    for i, category in enumerate(os.listdir(data_path)):
        if category not in labels.keys():
            continue
        category_path = os.path.join(data_path, category)
        for filename in os.listdir(category_path):
            filepath = os.path.join(category_path, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (512, 512))
            label = labels[category]
            training_data.append([img, label])

            img_count += 1

    print(f"TRAINING DATA: Loaded: {len(training_data)} files")
    print(f"shape: {training_data[0][0].shape} label: {training_data[0][1]}")

    return training_data


def main():
    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")
        os.mkdir("./dataset/images")

    data_path = "./raw_data"
    if not os.path.exists(data_path):
        assert "Please download the data"

    training_data_path = os.path.join(data_path, "Training")
    testing_data_path = os.path.join(data_path, "Testing")

    training_data = load_training_data(training_data_path)
    testing_data = load_testing_data(testing_data_path)

    with open("./dataset/training_data.pickle", "wb") as f:
        pickle.dump(training_data, f)
    with open("./dataset/testing_data.pickle", "wb") as f:
        pickle.dump(testing_data, f)


if __name__ == "__main__":
    main()
