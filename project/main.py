from src.utils import *
from src.crop_disease_dataset import *
from src.model import *
from src.train import *
from src.evaluation import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_dataset(extracted_folder, gray_scale=False, segmented=False):
    # Clone the repository
    repo_url = "https://github.com/digitalepidemiologylab/plantvillage_deeplearning_paper_dataset.git"
    clone_dir = "plantvillage_deeplearning_paper_dataset"
    clone_repo(repo_url, clone_dir)

    # Extract the desired folder from the cloned repository
    classes = extract_folder(repo_url, clone_dir, extracted_folder)

    # Plot a histogram of the class distribution
    plot_class_histogram(classes)

    # Load the datasets
    train_set = CropDiseaseDataset(root_dir=extracted_folder, train=True, validation=False, gray_scale=gray_scale, segmented=segmented)
    validation_set = CropDiseaseDataset(root_dir=extracted_folder, train=False, validation=True, gray_scale=gray_scale, segmented=segmented)
    test_set = CropDiseaseDataset(root_dir=extracted_folder, train=False, validation=False, gray_scale=gray_scale, segmented=segmented)

    trainloader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    validationloader = DataLoader(validation_set, batch_size=64, shuffle=False, num_workers=2)
    testloader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    # Display the sizes of the datasets
    print(f"Train set: '{len(train_set)}' images,", f"Validation set: '{len(validation_set)}' images,", f"Test set: '{len(test_set)}' images")

    # Define hyperparameters
    learning_rate = 1e-3
    num_epochs = 20

    # Initialize the network and optimizer
    network = CNN(gray_scale=gray_scale).to(device)
    network.apply(initialize_parameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    # Train the model
    train_avg_loss, validation_avg_loss, validation_accuracy, train_accuracy = train(network, num_epochs, trainloader, validationloader, criterion, optimizer, validation_phase=True)
    plot_training_results(train_avg_loss, validation_avg_loss, validation_accuracy, is_validation=True)

    # Combine training and validation datasets for testing the model
    combined_train_set = ConcatDataset([train_set, validation_set])
    combined_train_loader = DataLoader(combined_train_set, batch_size=64, shuffle=True, num_workers=2)

    # Test the model
    network = CNN(gray_scale=gray_scale).to(device)
    network.apply(initialize_parameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    train_avg_loss, test_avg_loss, test_accuracy, train_accuracy = train(network, num_epochs, combined_train_loader, testloader, criterion, optimizer, validation_phase=False)
    plot_training_results(train_avg_loss, test_avg_loss, test_accuracy, is_validation=False)

    # Save the model state with dynamic naming
    model_save_path = f'model_{extracted_folder.replace("/", "_")}.pt'
    torch.save(network.state_dict(), model_save_path)

    # Load the model state (for demonstration purposes)
    network.load_state_dict(torch.load(model_save_path))

    # Get model predictions
    images, labels, probs, corrects = get_predictions(network, testloader, device)
    pred_labels = torch.argmax(probs, 1)

    print(f"There are {len(corrects)} correct predictions.")

    # Plot confusion matrix
    plot_confusion_matrix(labels, pred_labels)

    # Extract and sort incorrect examples
    incorrect_examples = []
    for image, label, prob, correct in zip(images, labels, probs, corrects):
        if not correct:
            incorrect_examples.append((image, label, prob))

    incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)
    print(f"There are {len(incorrect_examples)} incorrect predictions.")

    incorrect_counts = count_incorrect_predictions(incorrect_examples)
    class_names = [train_set.classes[label] for label in incorrect_counts.keys()]
    plot_incorrect_predictions_histogram(incorrect_counts, class_names)

    data = {'Class Name': class_names, 'Incorrect Counts': list(incorrect_counts.values())}
    df1 = pd.DataFrame(data)
    df1 = df1.sort_values(by='Class Name')

    class_counts = defaultdict(int)

    for images, labels in testloader:
        for label in labels:
            class_counts[label.item()] += 1

    class_names_dict = {label: train_set.classes[label] for label in class_counts.keys()}
    class_counts_named = {class_names_dict[label]: count for label, count in class_counts.items()}

    plot_class_histogram(class_counts_named)

    df2 = pd.DataFrame(list(class_counts_named.items()), columns=['Class Name', 'Counts'])
    df2 = df2.sort_values(by='Class Name')

    merged_df = pd.merge(df1, df2, on='Class Name', suffixes=('_incorrect', '_test'))

    # Calculate the rate of success for each class in percentage
    merged_df['Success Rate (%)'] = ((merged_df['Counts'] - merged_df['Incorrect Counts']) / merged_df['Counts']) * 100
    merged_df = merged_df.sort_values(by='Class Name')

    N_IMAGES = 25
    plot_most_incorrect(incorrect_examples, N_IMAGES)

def main():
    # Process color, grey, and segmented images
    process_dataset(extracted_folder="raw/grayscale", gray_scale=True, segmented=False)
    process_dataset(extracted_folder="raw/color", gray_scale=False, segmented=False)
    process_dataset(extracted_folder="raw/segmented", gray_scale=False, segmented=True)

if __name__ == "__main__":
    main()
