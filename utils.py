import random 
from typing import Tuple, Literal
import math

from tqdm import tqdm
from datasets import Dataset, Image, ClassLabel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay

def calculate_mean_std(dataset, image_key = "image"):
    """
    Calculate mean and std of the dataset.
    Args:
        dataset: dataset object
    Returns:
        (mean, std): mean and std of the dataset
    """
    mean = 0.
    std = 0.
    for data in tqdm(dataset, "calculating mean and std"):
        image = data[image_key]
        if not image.is_floating_point():
            image = image.float() / 255.
        image = image.view(-1, image.size(-1))
        mean += image.mean(0)
        std += image.std(0)
    mean /= len(dataset)
    std /= len(dataset)
    return mean.tolist(), std.tolist()
    
def swap_labels(dataset, classes_to_swap: Tuple[Tuple[str, str]], swap_prob: Tuple[float], image_key = "image", label_key = "label", noisy_label_key = "noisy_label"):
    """
    Swap the labels in the dataset.
    Args:
        dataset: huggingface dataset object  
        labels_to_swap: list of tuples of classes to swap
                        e.g. [("cat", "dog"), ("bird", "airplane")]
        swap_prob: probability of swapping the labels
                   e.g. [0.1, 0.2] means 10% of the labels will be swapped with the first pair and 20% with the second pair
        label_key: key for the label in the dataset
        noisy_label_key: key for the noisy label in the dataset (binary label for swapped or not swapped)
    Returns:
        fake_dataset: dataset with swapped labels, with a new column for noisy labels
                      noisy label is 1 if the label is swapped, 0 otherwise
    """
    assert len(classes_to_swap) == len(swap_prob), "classes_to_swap and swap_prob must have the same length"

    classes = dataset.features[label_key].names
    for class1, class2 in classes_to_swap:
        assert class1 in classes, f"{class1} not in the dataset"
        assert class2 in classes, f"{class2} not in the dataset"
    
    labels_idxs = {}
    for i, row in enumerate(dataset):
        label_int = row[label_key]
        label = classes[label_int]
        labels_idxs.setdefault(label, []).append(i)
    
    dataset_dict = dataset.to_dict()
    dataset_dict["noisy_label"] = [0] * len(dataset)

    for (class1, class2), prob in zip(classes_to_swap, swap_prob):
        idxs1 = labels_idxs[class1]
        idxs2 = labels_idxs[class2]
        num_swap_1 = int(prob * len(idxs1))
        num_swap_2 = int(prob * len(idxs2))
        
        swapped_idxs_1 = random.sample(idxs1, num_swap_1)
        swapped_idxs_2 = random.sample(idxs2, num_swap_2)

        for idx in swapped_idxs_1:
            dataset_dict[label_key][idx] = classes.index(class2)
            dataset_dict[noisy_label_key][idx] = 1

        for idx in swapped_idxs_2:
            dataset_dict[label_key][idx] = classes.index(class1)
            dataset_dict[noisy_label_key][idx] = 1
    
    fake_dataset = Dataset.from_dict(dataset_dict).cast_column(image_key, Image()).cast_column(label_key, ClassLabel(num_classes=len(classes), names=classes))
    return fake_dataset.with_format("torch")

def plot_label_issue_examples(datalab, num_examples=15):
    column_names = datalab.data.column_names
    image_key, label_key, noisy_label_key = column_names
    noisy_labels = datalab.data[noisy_label_key]

    label_issues = datalab.get_issues(label_key)
    label_issues_df = label_issues.query("is_label_issue").sort_values("label_score")
    
    ncols = 5
    num_examples = min(num_examples, len(label_issues_df))
    nrows = int(math.ceil(num_examples / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(1.5 * ncols, 1.5 * nrows))
    axes_list = axes.flatten()
    label_issue_indices = label_issues_df.index.values
    
    for i, ax in enumerate(axes_list):
        if i >= num_examples:
            ax.axis("off")
            continue
        idx = int(label_issue_indices[i])
        row = label_issues.loc[idx]
        ax.set_title(
            f"id: {idx}\n GL: {row.given_label}\n SL: {row.predicted_label}\n NL: {noisy_labels[idx]}",
            fontdict={"fontsize": 8},
        )
        ax.imshow(datalab.data[idx][image_key], cmap="gray")
        ax.axis("off")
    plt.subplots_adjust(hspace=1)
    return fig
    
def plot_outlier_issues_examples(lab, num_examples = 15):
    column_names = lab.data.column_names
    image_key, label_key = column_names[0], column_names[1]

    label_issues = lab.get_issues(label_key)
    # label_issues_df = label_issues.query("is_label_issue").sort_values("label_score")

    outlier_issues_df = lab.get_issues("outlier")
    outlier_issues_df = outlier_issues_df.query("is_outlier_issue").sort_values("outlier_score")

    ncols = 4
    nrows = min(num_examples, len(outlier_issues_df))
    N_comparison_images = ncols - 1

    def sample_from_class(label, number_of_samples, index):
        index = int(index)

        non_outlier_indices = (
            label_issues.join(outlier_issues_df)
            .query("given_label == @label and is_outlier_issue.isnull()")
            .index
        )
        non_outlier_indices_excluding_current = non_outlier_indices[non_outlier_indices != index]

        sampled_indices = np.random.choice(
            non_outlier_indices_excluding_current, number_of_samples, replace=False
        )

        label_scores_of_sampled = label_issues.loc[sampled_indices]["label_score"]

        top_score_indices = np.argsort(label_scores_of_sampled.values)[::-1][:N_comparison_images]

        top_label_indices = sampled_indices[top_score_indices]

        sampled_images = [lab.data[int(i)][image_key] for i in top_label_indices]

        return sampled_images

    def get_image_given_label_and_samples(idx):
        image_from_dataset = lab.data[idx][image_key]
        corresponding_label = label_issues.loc[idx]["given_label"]
        comparison_images = sample_from_class(corresponding_label, 30, idx)[:N_comparison_images]

        return image_from_dataset, corresponding_label, comparison_images

    count = 0
    images_to_plot = []
    labels = []
    idlist = []
    for idx, row in outlier_issues_df.iterrows():
        idx = row.name
        image, label, comparison_images = get_image_given_label_and_samples(idx)
        labels.append(label)
        idlist.append(idx)
        images_to_plot.append(image)
        images_to_plot.extend(comparison_images)
        count += 1
        if count >= nrows:
            break

    ncols = 1 + N_comparison_images
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(1.5 * ncols, 1.5 * nrows))
    axes_list = axes.flatten()
    for i, ax in enumerate(axes_list):
        if i % ncols == 0:
            ax.set_title(f"id: {idlist[i // ncols]}\n GL: {labels[i // ncols]}", fontdict={"fontsize": 8})
        ax.imshow(images_to_plot[i], cmap="gray")
        ax.axis("off")
    plt.subplots_adjust(hspace=1)
    return fig

def plot_near_duplicate_issue_examples(lab, num_examples=15):
    column_names = lab.data.column_names
    image_key, label_key = column_names[0], column_names[1]

    label_issues = lab.get_issues(label_key)
    near_duplicate_issues_df = lab.get_issues("near_duplicate")
    near_duplicate_issues_df = near_duplicate_issues_df.query("is_near_duplicate_issue").sort_values(
        "near_duplicate_score"
    )

    # Determine the max number of duplicates to set subplot grid correctly
    max_duplicates = min(num_examples, near_duplicate_issues_df['near_duplicate_sets'].apply(lambda x: len(x)+1).max())

    temp = set()
    nrows = 0
    for idx, dup_set in near_duplicate_issues_df.near_duplicate_sets.items():
        dup_set = set(dup_set)
        if idx in temp or dup_set & temp: 
            continue
        temp.add(idx)
        temp.update(dup_set)
        nrows += 1

    nrows = min(num_examples, nrows)
    if nrows == 0:
        return None
    
    fig, axs = plt.subplots(nrows, max_duplicates, figsize=(max_duplicates * 1.5, nrows * 1.5))

    seen_id_pairs = []
    counter = 0  # For tracking the number of rows

    for idx, row in near_duplicate_issues_df.iterrows():
        if counter >= nrows:
            break

        duplicate_images = row['near_duplicate_sets']
        nd_set = [int(idx)] + [int(i) for i in duplicate_images]

        # Check if the current set has already been seen
        if any(x in seen_id_pairs for x in nd_set):
            continue  # Skip this set if any member has already been seen

        seen_id_pairs.extend(nd_set)

        for i in range(max_duplicates):
            # Adjust subplot access based on the number of rows
            if nrows > 1:
                ax = axs[counter, i]
            else:
                ax = axs[i]

            if i < len(nd_set):
                image, label = lab.data[nd_set[i]][image_key], label_issues.iloc[nd_set[i]]["given_label"]
                ax.imshow(image, cmap="gray")
                ax.set_title(f"id: {nd_set[i]}\nLabel: {label}", fontdict={"fontsize": 8})
            ax.axis("off")
        counter += 1

    plt.tight_layout()
    return fig

def plot_image_issue_examples(lab, num_examples=15, type: Literal["dark", "low_information"]="low_information"):
    column_names = lab.data.column_names
    image_key, label_key = column_names[0], column_names[1]

    label_issues = lab.get_issues(label_key)
    issues = lab.get_issues(type)
    issues_df = issues.query(f"is_{type}_issue").sort_values(f"{type}_score")
    
    ncols = 5
    num_examples = min(num_examples, len(issues_df))
    nrows = int(math.ceil(num_examples / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(1.5 * ncols, 1.5 * nrows))
    axes_list = axes.flatten()
    issue_indices = issues_df.index.values

    for i, ax in enumerate(axes_list):
        if i >= num_examples:
            ax.axis("off")
            continue
        idx = int(issue_indices[i])
        label = label_issues.loc[idx]["given_label"]
        predicted_label = label_issues.loc[idx]["predicted_label"]
        ax.set_title(
            f"id: {idx}\n GL: {label}\n SL: {predicted_label}",
            fontdict={"fontsize": 8},
        )
        ax.imshow(lab.data[idx][image_key], cmap="gray")
        ax.axis("off")

    plt.subplots_adjust(hspace=1)
    return fig

def roc_evaluation(lab):
    column_names = lab.data.column_names
    label_key, noisy_label_key = column_names[1], column_names[2]

    # label_score=1-probability of the label being a label issue
    # 1-label_score=probability of the label being a label issue (i.e. swapped label)
    pred_probs = 1-lab.get_issues(label_key).label_score
    # noisy_label is 1 if the label is swapped, 0 otherwise
    gts = lab.data[noisy_label_key]
    fpr, tpr, thresholds = roc_curve(gts, pred_probs)
    aucroc = auc(fpr, tpr)

    # Youden's J statistic to find the best threshold
    J = tpr-fpr
    J_max_idx = J.argmax()
    best_threshold = thresholds[J_max_idx]
    best_tpr = tpr[J_max_idx]
    best_fpr = fpr[J_max_idx]

    fig = plt.figure(1)
    plt.scatter(best_fpr, best_tpr, color='red', label=f"(fpr, tpr, thresh) = ({best_fpr:.2f}, {best_tpr:.2f}, {best_threshold:.2f})", marker="x")
    plt.plot((0, 1), (0, 1), 'k--', label="Random Guess (AUC = 0.5)")
    plt.plot(fpr, tpr, label=f"Classifier (AUC = {aucroc:.2f})")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    return aucroc, best_threshold, fig 

def label_issue_evaluation(lab, num_examples=50):
    """
    Evaluate the label issues and the classification metrics of the CL in identifying the noisy labels.
    It returns the classification report from sklearn.metrics.classification_report.
    Also, it plots the examples of FP and FN of the CL, i.e. the examples that the CL failed to identify as the noisy labels.
    Such plot contains the examples with label issues originally present in the dataset, 
    and the examples with synthetic label issues that the CL failed to identify.
    The examples are sorted by the label score in ascending order, i.e. the probability that the given label (GT) is correct.
    Only the first `num_examples` examples are plotted.

    Args:
    - lab: Datalab object
    - num_examples: int, the number of examples to be plotted

    Returns:
    - fig: matplotlib.figure.Figure, the figure containing the examples
    - conf_mat_fig: matplotlib.figure.Figure, the figure containing the confusion matrix
    - cls_report: dict, the classification report from sklearn.metrics.classification_report
    """
    column_names = lab.data.column_names
    image_key, label_key, noisy_label_key = column_names

    label_issues = lab.get_issues(label_key)
    label_issues[noisy_label_key] = lab.data[noisy_label_key]
    label_issues[noisy_label_key] = label_issues[noisy_label_key].astype(bool)
    # metrics of CL correctly identifying the noisy labels (synthetic noisy labels)
    preds_label_issues = label_issues["is_label_issue"]
    gts = label_issues[noisy_label_key]
    cls_report = classification_report(gts, preds_label_issues, output_dict=True)
    # inspect wrong predictions 
    # i.e. pred_CL=False(clean), noisy_label=True(noisy)
    #      or 
    #      pred_CL=True, noisy_label=False
    wrong_preds = label_issues.query(f"is_label_issue != {noisy_label_key}")
    wrong_preds = wrong_preds.sort_values("label_score")

    ncols = 5
    num_examples = min(num_examples, len(wrong_preds))
    nrows = int(math.ceil(num_examples / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(1.5 * ncols, 1.5 * nrows))
    axes_list = axes.flatten()
    label_issue_indices = wrong_preds.index.values
    
    for i, ax in enumerate(axes_list):
        if i >= num_examples:
            ax.axis("off")
            continue
        idx = int(label_issue_indices[i])
        row = label_issues.loc[idx]
        ax.set_title(
            f"id: {idx}\nGL: {row.given_label}\nSL: {row.predicted_label}\nNL: {label_issues[noisy_label_key][idx]}\nLS: {row.label_score:.2f}",
            fontdict={"fontsize": 8},
        )
        ax.imshow(lab.data[idx][image_key], cmap="gray")
        ax.axis("off")
    plt.subplots_adjust(hspace=1)
    
    conf_mat = confusion_matrix(label_issues["given_label"], label_issues["predicted_label"])
    disp = ConfusionMatrixDisplay(conf_mat, display_labels=lab.data.features[label_key].names)
    conf_mat_fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.tight_layout()
    return fig, conf_mat_fig, cls_report