# metapath2vec train code

import os
import platform
import argparse
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
import torch
from torch_geometric.datasets import AMiner
from torch_geometric.nn import MetaPath2Vec

from utils import *


# w1) start a new run
wandb.init(project="metapath2vec-train", entity="youyoung")

# w1) set run-name
ap = argparse.ArgumentParser(description="metapath2vec argparser")
ap.add_argument("--run-name", "-rn", type=str, default="new run", help="string for wandb run-name")
args = ap.parse_args()

wandb.run.name = args.run_name
wandb.run.save()


# load the data
EPS = 1e-15

root = 'data/AMiner'
path = os.path.join(os.getcwd(), root)
dataset = AMiner(path)
data = dataset[0]

metapath = [
    ('author', 'writes', 'paper'),
    ('paper', 'published_in', 'venue'),
    ('venue', 'publishes', 'paper'),
    ('paper', 'written_by', 'author'),
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# get params from ini file
parser = ConfigParser()
_ = parser.read('config.ini')

params = {
    'embedding_dim': int(parser.get('setting', 'embedding_dim')),
    'walk_length': int(parser.get('setting', 'walk_length')),
    'context_size': int(parser.get('setting', 'context_size')),
    'walks_per_node': int(parser.get('setting', 'walks_per_node')),
    'num_negative_samples': int(parser.get('setting', 'num_negative_samples'))
}

LR = 1e-2

# w2) save model parameters
wandb.config = params
wandb.learning_rate = LR

model = MetaPath2Vec(
    data.edge_index_dict,
    embedding_dim=params['embedding_dim'],
    metapath=metapath,
    walk_length=params['walk_length'],
    context_size=params['context_size'],
    walks_per_node=params['walks_per_node'],
    num_negative_samples=params['num_negative_samples'],
    sparse=True).to(device)

# w3) watch model
wandb.watch(model)

# pytorch multiprocessing does not work in windows
num_workers = 0 if platform.system() == 'Windows' else os.cpu_count()
loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)

# lazy version of Adam algorithm suitable for sparse tensors.
# In this variant, only moments that show up in the gradient get updated,
# and only those portions of the gradient get applied to the parameters.
optimizer = torch.optim.SparseAdam(
    list(model.parameters()), lr=LR, betas=(0.9, 0.999), eps=1e-08)

EPOCHS = 5


# define train/test code
def train(epoch):
    model.train()
    total_loss = 0
    loop = tqdm(enumerate(loader), total=int(len(loader)))

    for i, (pos_rw, neg_rw) in loop:
        loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")

        # back-propagation
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        mean_loss = total_loss / (i+1)
        loop.set_postfix(loss=mean_loss)

        # w4) log
        wandb.log({'mean loss per epoch': mean_loss})


@torch.no_grad()
def test(train_ratio=0.1):
    # returns train/test score after fitting Logistic Regression model
    model.eval()

    # changing device type is needed (to(device))
    z = model('author', batch=data['author'].y_index.to(device))
    y = data['author'].y    # author id

    perm = torch.randperm(z.size(0))
    train_perm = perm[:int(z.size(0) * train_ratio)]
    test_perm = perm[int(z.size(0) * train_ratio):]

    return model.test(
        z[train_perm], y[train_perm], z[test_perm], y[test_perm], max_iter=150)


def visualize_venue_emb_vec(epoch):
    # get venue embedding vector and apply tsne
    z = model('venue', batch=data['venue'].y_index.to(device)).detach().cpu().numpy()
    tsne = TSNE(n_components=2, verbose=0, learning_rate='auto', init='random', random_state=0)
    z = tsne.fit_transform(z)[:133, :]

    # get venue label info
    path = os.path.join(os.getcwd(), 'data/AMiner/raw/label/googlescholar.8area.venue.label.txt')
    df = pd.read_csv(path, sep=' ', names=['name', 'y'])
    df = df.rename({'y': 'label'}, axis=1)

    # venue_label = pd.DataFrame({'label': data['venue'].y, 'y_index': data['venue'].y_index})
    # venue_label['label'] += 1
    # venue_label = venue_label.head(df.shape[0])
    # df['y_index'] = venue_label['y_index']

    df['x_coor'], df['y_coor'] = z[:, 0], z[:, 1]

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        x="x_coor", y="y_coor", hue=df.label.tolist(), legend="brief",
        palette=sns.color_palette("hls", 8),
        data=df).set(title=f"Vis of venue emb_vec: epoch{epoch}")

    # w4) log
    wandb.log({'plot': wandb.Image(fig)})


def main():
    for epoch in range(1, EPOCHS+1):
        train(epoch)
        acc = test()
        # w4) log
        wandb.log({'test acc per epoch': acc})
        print(f'Test Accuracy after epoch {epoch}: {acc:.4f}')

        visualize_venue_emb_vec(epoch)


if __name__ == '__main__':
    main()
