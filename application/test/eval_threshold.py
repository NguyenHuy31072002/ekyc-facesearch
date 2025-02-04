import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pandas as pd


def visualize(angles_file, labels, threshold):
    df = pd.read_csv(angles_file, header=0)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    bins = np.linspace(0, 1, 100)

    for i, l in enumerate(labels):
        scores = df['score'][df['label'] == l].tolist()

        plt.hist(scores, bins, density=False, alpha=0.5, label=l, facecolor=colors[i])

        mu_0 = np.mean(scores)
        sigma_0 = np.std(scores)
        y_0 = scipy.stats.norm.pdf(bins, mu_0, sigma_0)
        plt.plot(bins, y_0, color=colors[i], linestyle='--')

        print(f'mu_{l}: {mu_0}')
        print(f'sigma_{l}: {sigma_0}')

    plt.xlabel('theta')
    plt.ylabel('theta j Distribution')
    # plt.title(
    #     r'Histogram : mu_0={:.4f},sigma_0={:.4f}, mu_1={:.4f},sigma_1={:.4f}'.format(mu_0, sigma_0, mu_1, sigma_1))

    print('threshold: ' + str(threshold))

    plt.legend(loc='upper right')
    plt.plot([threshold, threshold], [0, 10], 'k-', lw=2)
    plt.savefig('theta_dist.png')
    plt.show()


def get_threshold(angles_file):
    df = pd.read_csv(angles_file, header=0)
    min_error = len(df)
    min_threshold = 0

    for idx, row in df.iterrows():
        threshold = row['score']
        type1 = len([s for i, s in df.iterrows() if s['score'] <= threshold and s['label'] == 'fake'])
        type2 = len([s for i, s in df.iterrows() if s['score'] > threshold and s['label'] == 'real'])
        num_errors = type1 + type2
        if num_errors < min_error:
            min_error = num_errors
            min_threshold = threshold

    print(f'Min error = {min_error},  Min threshold = {min_threshold}')
    return min_threshold


def accuracy(file, threshold):
    df = pd.read_csv(file, header=0)
    wrong = 0
    for idx, row in df.iterrows():
        if (row['score'] <= threshold and row['label'] == 'fake') or (
                row['score'] > threshold and row['label'] == 'real'):
            wrong += 1

    accuracy = 1 - wrong / len(df)
    print(f'ACC = {accuracy}')
    return accuracy


def error_analysis(file, threshold):
    df = pd.read_csv(file, header=0)
    fp = 0
    fn = 0
    for idx, row in df.iterrows():
        if (row['score'] <= threshold and row['label'] == 'fake'):
            fn += 1
        if (row['score'] > threshold and row['label'] == 'real'):
            fp += 1

    print(f'FP = {fp}; FN = {fn}')


if __name__ == '__main__':
    predict_file = '/media/thiennt/projects/remote_lvt/ekyc-lvt/application/modules/face/liveness/predict_score.txt'
    threshold = 0.7
    threshold = get_threshold('/media/thiennt/projects/remote_lvt/ekyc-lvt/application/modules/face/liveness/predict_score.txt')
    accuracy(predict_file, threshold)
    error_analysis(predict_file, threshold)
    visualize(predict_file,
              ['real', 'fake'], threshold)
