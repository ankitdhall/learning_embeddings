import numpy as np
import matplotlib
import matplotlib.pyplot as plt



N_leaves, N_family, N_subfamily = 561, 6, 21

from data.db import ETHECLabelMapMerged

np.random.seed(0)

# predicted_scores = np.random.uniform(0, 1, (N_samples, N_leaves))
# correct_labels = np.random.uniform(0, 1, (N_samples, N_leaves))

def plot(predicted_level=3, gt_level=0):
    labelmap = ETHECLabelMapMerged()

    predicted_scores = np.load('predicted_scores4x1.npy')
    correct_labels = np.load('correct_labels4x1.npy')

    correct_labels_full = correct_labels

    predicted_scores = predicted_scores[:, labelmap.level_start[predicted_level]:labelmap.level_stop[predicted_level]]
    correct_labels = correct_labels[:, labelmap.level_start[predicted_level]:labelmap.level_stop[predicted_level]]

    predicted_labels = np.zeros_like(predicted_scores)
    predicted_labels[[i for i in range(predicted_scores.shape[0])], np.argmax(predicted_scores, axis=1)] = 1

    per_label_acc, weights = [], []
    for ix in range(predicted_labels.shape[1]):
        per_label_acc.append(predicted_labels[np.where(correct_labels[:, ix] == 1), ix].mean())
        weights.append(1.0/np.where(correct_labels[:, ix] == 1)[0].shape[0])
    weights = np.array(weights)
    weights = weights/np.sum(weights, keepdims=True)
    print('weighted accuracy: {}'.format(np.sum(per_label_acc*weights)))

    if gt_level < predicted_level:
        x_axis = gt_level
        y_axis = predicted_level
    else:
        x_axis = predicted_level
        y_axis = gt_level

    cmat = np.zeros((labelmap.levels[x_axis], labelmap.levels[y_axis]))

    print(cmat.shape)
    print(predicted_scores.shape, predicted_labels.shape)

    for sample_id in range(predicted_labels.shape[0]):
        if gt_level < predicted_level:
            cmat[np.argmax(correct_labels_full[sample_id, labelmap.level_start[x_axis]:labelmap.level_stop[x_axis]]), np.where(predicted_labels[sample_id, :] == 1)] += 1
        else:
            cmat[np.where(predicted_labels[sample_id, :] == 1), np.argmax(correct_labels_full[sample_id, labelmap.level_start[y_axis]:labelmap.level_stop[y_axis]])] += 1

    print(cmat)
    cmat_sorted = np.zeros_like(cmat)
    # get all possible child labels at level x_axis for labels in y_axis

    children = {}
    for label_id in range(labelmap.levels[x_axis]):
        # print('Looking for children of {} from level {}'.format(label_id, y_axis))
        children[label_id] = [label_id]
        for level_id in range(x_axis+1, y_axis+1):
            children_in_next_level = []
            for child_label in children[label_id]:
                children_in_next_level.extend(labelmap.get_children_of(child_label, level_id))
                # print('Children of {} living in {} are {}'.format(child_label, level_id, labelmap.get_children_of(child_label, level_id)))
            # print('Children of {} living in {} are {}'.format(label_id, level_id, children_in_next_level))
            children[label_id] = children_in_next_level

    print(children)
    rearrange_ix = 0
    sorting_indices = []
    for parent_id in children:
        cmat_sorted[:, rearrange_ix:rearrange_ix+len(children[parent_id])] = cmat[:, children[parent_id]]
        rearrange_ix += len(children[parent_id])
        sorting_indices.extend(children[parent_id])
    # print(rearrange_ix)

    # print([getattr(labelmap, '{}_ix_to_str'.format(labelmap.level_names[x_axis]))[label_ix] for label_ix in sorting_indices])

    cmat = cmat_sorted

    # print(cmat.sum(axis=0, keepdims=True))
    if gt_level < predicted_level:
        cmat = cmat/cmat.sum(axis=1, keepdims=True)
    else:
        cmat = cmat / cmat.sum(axis=0, keepdims=True)
    cmat[np.isnan(cmat)] = 0

    cmap = plt.cm.gist_stern_r #cividis
    tick_names = [getattr(labelmap, '{}_ix_to_str'.format(labelmap.level_names[y_axis]))[label_ix] for label_ix in sorting_indices]

    fig, ax = plt.subplots()
    im = ax.imshow(cmat, interpolation='nearest', cmap=cmap, norm=matplotlib.colors.LogNorm())
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(#xticks=np.arange(cmat.shape[1]),
           # xticks=np.arange(len(tick_names)),
           # xticklabels=tick_names,
           # ... and label them with the respective list entries
           # xticklabels=classes, yticklabels=classes,
           # title=title,
           ylabel='{} {}'.format('Predicted' if x_axis == predicted_level else 'True', labelmap.level_names[x_axis]),
           xlabel='{} {}'.format('Predicted' if y_axis == predicted_level else 'True', labelmap.level_names[y_axis]),
           aspect=10)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    fig.tight_layout()
    plt.show()


plot(predicted_level=0, gt_level=3)
