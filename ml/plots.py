"""Static EDA from measured data, not illustrative charts."""
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_eda(histories, model, games, audits, test_cases, path):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    histories_count = Counter(map(len, histories.values()))
    axes[0, 0].bar(list(histories_count), list(histories_count.values()))
    axes[0, 0].set(xlabel='Positive interactions / user', ylabel='Users (log)', yscale='log', title='Observed cleaned histories (sample)')
    axes[0, 1].hist(model.popularity, bins=np.geomspace(1, max(model.popularity)+1, 30))
    axes[0, 1].set(xlabel='Train positive users / game (log)', ylabel='Games (log)', xscale='log', yscale='log', title='Train item popularity')
    tags = Counter(tag for item in model.catalog for tag in games[item]['tags']).most_common(10)
    axes[0, 2].barh([t[0] for t in tags][::-1], [t[1] for t in tags][::-1])
    axes[0, 2].set(xlabel='Train catalog games', title='Ten most frequent tags')
    labels, before, after = [], [], []
    for part in ('validation', 'test'):
        for k in (1, 3, 5):
            a = audits[part][f'common/{k}']
            labels.append(f'{part[:3]} k={k}')
            before.append(a['eligible_history'])
            after.append(a['included_users'])
    x = np.arange(len(labels))
    axes[1, 0].bar(x-.2, before, .4, label='Before catalog filter')
    axes[1, 0].bar(x+.2, after, .4, label='Included')
    axes[1, 0].set(xticks=x, xticklabels=labels, ylabel='Users', title='Common cohort retention')
    axes[1, 0].legend()
    targets = [model.popularity[model.index[item]] for case in test_cases for item in case['query']]
    axes[1, 1].hist(targets, bins=np.geomspace(1, max(model.popularity)+1, 20))
    axes[1, 1].set(xlabel='Target train popularity (log)', ylabel='Query occurrences', xscale='log', title='Test-development common query targets')
    counts = [sum(bool(games[item].get(field)) for item in model.catalog) for field in ('tags', 'genres')]
    axes[1, 2].bar(['tags', 'genres'], counts)
    axes[1, 2].set(ylabel='Games with nonempty metadata', ylim=(0, len(model.catalog)*1.1), title=f'Metadata coverage / {len(model.catalog)} train items')
    fig.suptitle('UCSD Steam V1 reviews + V2 metadata | M1, seed 42 | measured sample statistics')
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.text(.5, .015, 'Source: pinned UCSD files; sample and cohort filters in manifest. Historical review data, not all Steam players.', ha='center')
    fig.savefig(path, dpi=150)
    plt.close(fig)
