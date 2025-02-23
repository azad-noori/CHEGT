import argparse
import sys


argv = sys.argv
#dataset = argv[1]
dataset = "acm"  # می‌توانید این مقدار را به دیتاست مورد نظر تغییر دهید

def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])  # داده های آموزشی
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=5)  # با مقدار 5 نتایج بهتری بدست آمد
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=1000)
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[7, 1])
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.3)

    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # تعداد نودهای هر نوع
    args.nei_num = 2  # تعداد انواع همسایه‌ها
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=65)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=1000)
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_drop', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.35)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[6])
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.2)

    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # تعداد نودهای هر نوع
    args.nei_num = 1  # تعداد انواع همسایه‌ها
    return args


def imdb_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=1000)
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--sample_rate', nargs='+', type=int, default=[3, 5])
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.2)

    args, _ = parser.parse_known_args()
    args.type_num = [5000, 2000, 1000]  # تعداد نودهای هر نوع
    args.nei_num = 2  # تعداد انواع همسایه‌ها
    return args


def set_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == "imdb":
        args = imdb_params()
    return args