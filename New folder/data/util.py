import os
import glob
import random
from PIL import Image
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate, learning_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter, defaultdict

# Classes ordered as Kaggle expects them
CLASSES = ['A13', 'A14', 'A15', 'A1AB', 'A1CD', 'A23', 'A23_yellow', 'A25', 'A29', 'A31', 'A51', 'A7A', 'A7B', 'B1', 'B11', 'B15A', 'B17', 'B19', 'B21', 'B3', 'B5', 'B7', 'B9', 'C1', 'C11', 'C21', 'C23', 'C29', 'C3', 'C31', 'C35', 'C37', 'C43', 'D10', 'D1a', 'D1b', 'D1e', 'D5', 'D7', 'D9', 'E1', 'E3', 'E5', 'E7', 'E9a', 'E9a_miva', 'E9b', 'E9cd', 'E9e', 'F1', 'F12a', 'F12b', 'F13', 'F19', 'F1a_h', 'F21', 'F23A', 'F25', 'F27', 'F29', 'F31', 'F33_34', 'F35', 'F3a_h', 'F41', 'F43', 'F45', 'F47', 'F49', 'F4a', 'F4b', 'F50', 'F59', 'F87', 'Handic', 'X', 'begin', 'e0c', 'end', 'lang', 'm']

def _split_path(path):
    """Splits a path in an OS-independent manner."""
    path = os.path.normpath(path)
    return path.split(os.sep)

# Data loaders
class TestDataLoader(object):
    """Data loader class for the test set.
    You can use this class to load the entire test set into memory.
    See the `load` function."""
    def __init__(self, root_path):
        """Initialize the data loader.
        The root path indicates the path to the `data` folder."""
        self.root_path = root_path
        # Note the call to sorted,
        # we want the test samples to be loaded according to the numeric value in their filenames,
        # so that they are in the correct order for Kaggle.
        self.all_samples = sorted(glob.glob(os.path.join(self.root_path, 'test', '*.png')), key=lambda s: int(_split_path(s)[-1].replace('.png', '')))
    
    def load(self):
        """Loads the entire test set into memory.
        Returns a list of samples, where each sample is a numpy array containing the RGB pixels as loaded by PIL.

        To obtain the original image from the array, simply call PIL.Image.fromarray(sample).
        """
        X = []
        for sample in self.all_samples:
            X.append(np.asarray(Image.open(sample)))
        return X
    
    def __len__(self):
        return len(self.all_samples)

class TrainDataLoader(object):
    """Data loader class for the training set.
    You can use this class to load the entire training set into memory.
    See the `load` function."""
    def __init__(self, root_path):
        """Initialize the data loader.
        The root path indicates the path to the `data` folder."""
        self.root_path = root_path
        self.all_samples = glob.glob(os.path.join(self.root_path, 'train', '*', '*', '*.png'))
    
    def load(self):
        """Loads the entire training set into memory.
        Returns a tuple containing following lists:

        - X: the RGB pixels as loaded by PIL
        - superclass: The super-class (diamonds, stop, etc.)
        - y: The class label (B1, D1a, etc.)
        - pole: The pole ID
        - index: The index of the sample

        Note that a sample is *not* uniquely defined by the index: there are several samples
        that share the same index but have, e.g., a different pole ID.
        """
        
        X, superclass, y, pole, index = [], [], [], [], []
        for sample in self.all_samples:
            split = _split_path(sample)
            X.append(np.asarray(Image.open(sample)))
            superclass.append(split[-3])
            y.append(split[-2])
            sample_split = split[-1].split('_')
            pole.append(sample_split[0])
            index.append(sample_split[1])
        return X, superclass, y, pole, index
    
    def __len__(self):
        return len(self.all_samples)

# Submission
def create_submission(probabilities, path):
    """Creates a submission file on the given path.

    :param probabilities: The output of estimator.predict_proba(); the order should correspond to classes  
    :param path: The path to the output CSV file  
    """
    p = probabilities.tolist()
    with open(path, 'w') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow(['Id', *CLASSES])
        for i in range(len(p)):
            writer.writerow([i + 1, *[str(e) for e in p[i]]])

# Other helpful functions
def stratified_group_k_fold(X, y, groups, k, seed=45):
    """Custom cross validation method
    Stratified: folds are made by preserving the percentage of samples for each class. 
    Grouped: The same group will not appear in two different folds.
    
    :param X: An array of shape (n_samples, n_features)  
    :param y: An array of shape (n_samples,)  
    :return: train and test indices
    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

def neg_logloss_scorer(estimator, X, y):
    """Scores the predictions of given estimator on X against y.

    Since scikit-learn's default behaviour is to maximize a score function,
    returns negative log loss.

    :param estimator: A scikit-learn estimator with support for predict_proba()  
    :param X: An array of shape (n_samples, n_features)  
    :param y: An array of shape (n_samples,)  
    :return: The negative log loss
    """
    prob = estimator.predict_proba(X)
    loss = log_loss(y, prob, labels=estimator.classes_) # returns negative log loss (minimize)
    return -loss # returns positive log loss (maximize)

def plot_learning_curve(learning_curve_result, title):
    """Plots a learning curve that you have pre-computed. 
    Training and cross validation scores in function of the amount of samples:
    important for the BIAS and VARIANCE of the model!
    
    :param learning_curve_result: The result of a call to learning_curve(...)
    :param title: Title of the resulting plot
    """
    train_sizes, train_scores, test_scores = learning_curve_result

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig = plt.figure()
    ax = fig.gca()

    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1,
                    color="g")
    
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
        label="Cross-validation score")
    ax.set_ylabel('Score')
    ax.legend(loc="best")
    ax.set_xlabel('Number of training samples')
    ax.set_title(title)
    save_path = os.path.join("visualization", title)
    fig.savefig(save_path)

def plot_confusion_matrix(confusion_matrix_result, title):
    """Plots a confusion matrix that you have pre-computed. 
    Is the model confusing two classes? I.e. the model mislables a as b and vice versa:
    false positives, false negatives, ...
    
    :param confusion_matrix_result: The result of a call to confusion_matrix(...)
    :param title: Title of the resulting plot
    """
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(title)
    ax.axis('off')

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result)
    
    disp.plot(include_values=False, ax=ax)
    save_path = os.path.join("visualization", title)
    fig.savefig(save_path)

# Our functions 
def countOcc(l):
    l = list(l)
    count = []
    ele = []
    for elem in set(l):
        count.append(l.count(elem))
    count.sort()
    count.reverse()
    return count

def apply_contrast(image):
    """
    Apply contrast enhancement using CLAHE.
    :param image: Array of (h,w,3) with h=height and w=width 
    """
    # go to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    
    # create clahe object and apply to lightness (L) component
    # clipLimit: contrast limit
    # tileGridSize: enhance contrast per 8x8 pixels
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    
    # go back to RGB color space
    return cv2.cvtColor(cv2.merge(lab_planes), cv2.COLOR_LAB2BGR)

def reshape_keep(image, s=64):
    """
    Reshapes an image to a predetermined height or width, keeping aspect ratio.
    If height > width, resize height to s;
    If height < width, resize width to s;
    Interpolation method left to default value of linear interpolation.
    :param image: Array of (h,w,3) with h=height and w=width 
    :returns: Array of (s,w,3) if h > s or (h,s,3) if h < s 
    """
    #print(image.shape)
    height, width, rgb = image.shape
    if height > width:
        new_width = int(width / height * s) # keep aspect ratio
        return cv2.resize(image, (new_width,s))
    elif height < width:
        new_height = int(height / width * s) # keep aspect ratio
        return cv2.resize(image, (s,new_height))
    else:
        return cv2.resize(image, (s,s))
    return cv2.resize(image, (s, s), interpolation=cv2.INTER_LINEAR)

def add_border(image, s=64, border_val=0):
    """
    Adds a border to an image to make it size (s,s). 
    It is expected that the input image has size (s,w,3) with w < s (h,s,3) with h < s.
    This can be done by the reshape_keep function above.
    :param image: array of (s,w,3) or array of (w,s,3)
    :param s: predetermined size of image s
    :param border_val: color of border (default black)
    :returns: array of (s,s,3)
    """
    height, width, rgb = image.shape
    #print(height, width)
    if height > width: 
        border_left = int( (s - width) // 2) # add border to width
        border_right = s - border_left - width
        return cv2.copyMakeBorder(
            image, 0, 0, border_left, border_right, cv2.BORDER_CONSTANT, border_val)
    elif height < width:
        border_top = int( (s - height) // 2) # add border to height
        border_bot = s - border_top - height
        return cv2.copyMakeBorder(
            image, border_top, border_bot, 0, 0, cv2.BORDER_CONSTANT, border_val)
    else: # image already has predetermined size
        return image

def reshape_(image, s=64):
    """
    Reshapes an image to y x y resolution. Interpolation method 
    left to default value of linear interpolation.
    :param image: Array of (h,w,3) with h=height and w=width 
    :returns: Array of (y,y,3) 
    """
    return cv2.resize(image, (s, s), interpolation=cv2.INTER_LINEAR)

def plot_histogram(axis, histogram, bins):
    """Plots a single histogram on a given axis.

    :param axis: A matplotlib axis object  
    :param histogram: A histogram loaded from the provided JSON files  
    :param color_name: One of 'red', 'green' and 'blue'  
    """
    axis.hist(histogram, bins)

def hist_3d_(data,nbin=8):
    hist = np.zeros((nbin**3,data.shape[0]))
    for i,image in enumerate(data):
        for x in np.arange(image.shape[0]):
            for y in np.arange(image.shape[1]):
                R = int(image[x,y,0]/(256/nbin))
                G = int(image[x,y,1]/(256/nbin))
                B = int(image[x,y,2]/(256/nbin))
                hist[int(R*nbin**2+G*nbin+B),i]+=1
        hist[:,i] = hist[:,i]/np.sum(hist[:,i])
    return hist

def image_hist_plot(hist,image,colors):
    plt.figure(figsize=(20,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(hist)),hist,width=1,color=colors,linewidth=0)

def plot_correlation_matrix(df, number_of_ticks=50):
    """plot the correlation matrix between different columns of a pandas dataframe
    
    :param df: pandas dataframe
    :param number_of_ticks: number of labels showing on the plot
    """
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize = (14,10))
    plt.jet() # set the colormap to jet
    cax = ax.matshow(df.corr(), vmin=-1, vmax=1)

    ticks = [i for i in range(0, len(df.columns), len(df.columns)//number_of_ticks)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(df.columns[ticks], rotation=90, horizontalalignment='left')
    ax.set_yticklabels(df.columns[ticks])
    
    fig.colorbar(cax, ticks=[-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0])

    plt.tight_layout()
    plt.show()

def show_scores(scores):
    """Show the training and cross-validation scores of a model.
    
    :param scores: scores object calculated using the cross_validate function
    """
    train_score_mean = np.mean(scores['train_score'])
    train_score_std = np.std(scores['train_score'])
    cv_score_mean = np.mean(scores['test_score'])
    cv_score_std = np.std(scores['test_score'])

    print('Training score {} +/- {}'.format(train_score_mean, train_score_std))
    print('Cross-validation score: {} +/- {}'.format(cv_score_mean, cv_score_std))

def show_grid_scores(grid_search):
    """Show the training and cross-validation scores for a gridsearch of a model.
    
    :param grid_search: grid_search object using the GridSearchCV function
    """
    print("Grid scores on training data set:")
    cv_means = grid_search.cv_results_['mean_test_score']
    cv_stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(cv_means, cv_stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    # display best parameter set
    print("Best parameters set found on development set: ", grid_search.best_params_)
    print('')

def get_best_model(grid_search):
    """Returns the best model of a grid search
    
    :param grid_search: grid_search object using the GridSearchCV function
    """
    estimator = grid_search.best_estimator_
    print('Obtained pipeline:')
    print(estimator)
    return estimator
